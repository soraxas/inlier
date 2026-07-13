//! Gallery demo: essential matrix estimation from calibrated 2D-2D correspondences.
//!
//! Shows two-camera geometry with frustums, shared 3D scene points, and
//! epipolar lines computed from the known ground-truth pose.
//!
//! Since spatialrust-inlier does not yet have an essential-matrix estimator,
//! this demo uses the analytic ground-truth pose to demonstrate the geometry.
//! It is wired up as a gallery plugin following the same pattern as the other
//! demos in this crate.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use crate::AppDemo;
use crate::algo_config::{RansacAlgo, algo_combo_ui, randomize_button_ui};

// ── Plugin / State / Marker ───────────────────────────────────────────────────

pub struct EssentialMatrixPlugin;

impl Plugin for EssentialMatrixPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EssentialMatrixState>()
            .add_systems(OnEnter(AppDemo::EssentialMatrix), on_enter)
            .add_systems(OnExit(AppDemo::EssentialMatrix), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                essential_matrix_ui.run_if(in_state(AppDemo::EssentialMatrix)),
            )
            .add_systems(
                Update,
                essential_matrix_scene.run_if(in_state(AppDemo::EssentialMatrix)),
            );
    }
}

#[derive(Component)]
pub struct EssentialMatrixEntity;

#[derive(Resource)]
pub struct EssentialMatrixState {
    pub show_epipolar: bool,
    pub show_3d_points: bool,
    pub n_inliers: usize,
    pub n_outliers: usize,
    pub sampson_threshold: f32,
    pub seed: u64,
    pub algo: RansacAlgo,
    pub needs_run: bool,
    // Results
    pub n_inliers_result: usize,
    pub rotation_angle_deg: f32,
    pub translation_norm: f32,
    // Confusion matrix
    pub conf_tp: usize,
    pub conf_fp: usize,
    pub conf_tn: usize,
    pub conf_fn: usize,
}

impl Default for EssentialMatrixState {
    fn default() -> Self {
        Self {
            show_epipolar: true,
            show_3d_points: true,
            n_inliers: 60,
            n_outliers: 18,
            sampson_threshold: 0.01,
            seed: 0xE551_EE55_u64,
            algo: RansacAlgo::default(),
            needs_run: true,
            n_inliers_result: 0,
            rotation_angle_deg: 0.0,
            translation_norm: 0.0,
            conf_tp: 0,
            conf_fp: 0,
            conf_tn: 0,
            conf_fn: 0,
        }
    }
}

// ── System: enter / exit ──────────────────────────────────────────────────────

fn on_enter(mut state: ResMut<EssentialMatrixState>) {
    state.needs_run = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<EssentialMatrixEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

// ── System: UI ────────────────────────────────────────────────────────────────

fn essential_matrix_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<EssentialMatrixState>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("essential_matrix_panel")
        .default_size(260.0)
        .show(ctx, |ui| {
            ui.heading("Essential Matrix");
            ui.separator();
            ui.label("2-camera geometry from calibrated correspondences.");
            ui.separator();

            if algo_combo_ui(ui, &mut state.algo) { state.needs_run = true; }

            ui.separator();
            let c1 = ui.checkbox(&mut state.show_epipolar, "Show epipolar lines").changed();
            let c2 = ui.checkbox(&mut state.show_3d_points, "Show 3D points").changed();
            if c1 || c2 {
                state.needs_run = true;
            }

            ui.separator();
            if ui
                .add(egui::Slider::new(&mut state.n_inliers, 5..=400).text("n_inliers"))
                .changed()
            {
                state.needs_run = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.n_outliers, 0..=300).text("n_outliers"))
                .changed()
            {
                state.needs_run = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.sampson_threshold, 0.0001..=0.5)
                    .text("Sampson the.")
                    .logarithmic(true))
                .changed()
            {
                state.needs_run = true;
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Re-run").clicked() { state.needs_run = true; }
                if randomize_button_ui(ui, &mut state.seed) { state.needs_run = true; }
            });

            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Pose");
            ui.label(format!("Rotation:   {:.2}°", state.rotation_angle_deg));
            ui.label(format!("Trans norm: {:.4}", state.translation_norm));

            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_BLUE, "Confusion (Sampson < the → inlier)");
            egui::Grid::new("em_confusion").num_columns(2).show(ui, |ui| {
                ui.colored_label(egui::Color32::GREEN, format!("TP {:3}", state.conf_tp));
                ui.colored_label(egui::Color32::RED,   format!("FP {:3}", state.conf_fp));
                ui.end_row();
                ui.colored_label(egui::Color32::from_rgb(100, 180, 255), format!("FN {:3}", state.conf_fn));
                ui.colored_label(egui::Color32::GRAY,  format!("TN {:3}", state.conf_tn));
                ui.end_row();
            });
            let tp = state.conf_tp; let fp = state.conf_fp;
            let fn_ = state.conf_fn;
            if tp + fp > 0 {
                let prec = tp as f32 / (tp + fp) as f32;
                let col = if prec > 0.9 { egui::Color32::GREEN } else if prec > 0.7 { egui::Color32::YELLOW } else { egui::Color32::RED };
                ui.colored_label(col, format!("Precision: {:.1}%", prec * 100.0));
            }
            if tp + fn_ > 0 {
                let rec = tp as f32 / (tp + fn_) as f32;
                let col = if rec > 0.9 { egui::Color32::GREEN } else if rec > 0.7 { egui::Color32::YELLOW } else { egui::Color32::RED };
                ui.colored_label(col, format!("Recall:    {:.1}%", rec * 100.0));
            }

            ui.separator();
            ui.small("Blue frustum = cam1   Orange = cam2");
            ui.small("White line   = baseline");
            ui.small("Gray cubes   = 3D scene points");
            ui.small("Colored lines = epipolar lines (first 5)");
        });
}

// ── System: scene ─────────────────────────────────────────────────────────────

fn essential_matrix_scene(
    mut commands: Commands,
    mut state: ResMut<EssentialMatrixState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<EssentialMatrixEntity>>,
) {
    if !state.needs_run {
        return;
    }
    state.needs_run = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    // ── Ground-truth pose ─────────────────────────────────────────────────────
    // Cam2 relative to cam1.
    let gt_rot = euler_to_rot(0.03, -0.08, 0.02);
    let gt_t_raw = [0.25_f32, -0.04, 0.03];
    let gt_t = normalize3(gt_t_raw);

    // Store stats
    let rot_angle = rot_to_angle_deg(&gt_rot);
    let t_norm = len3(gt_t_raw);
    state.rotation_angle_deg = rot_angle;
    state.translation_norm = t_norm;

    // ── Generate 3D scene points (LCG seed 0xE551_EE55) ──────────────────────
    let n_inlier_pts = state.n_inliers.max(2);
    let n_outlier_pts = state.n_outliers;
    state.n_inliers_result = n_inlier_pts;

    let scene_pts = gen_scene_points(n_inlier_pts, state.seed);

    // ── Project to cam1 and cam2 (focal = 1, K = I) ──────────────────────────
    // Cam1 centre at origin, R1 = I.
    let c1 = [0.0_f32, 0.0, 0.0];
    let r1 = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Cam2 centre: C2 = -R2^T * t_cam2_in_world,  but we define R2 = gt_rot (cam2 orientation),
    // and translation t as the baseline in world coords, so C2 = c1 + t (scaled).
    let baseline_scale = 2.0_f32;
    let c2 = [
        c1[0] + gt_t[0] * baseline_scale,
        c1[1] + gt_t[1] * baseline_scale,
        c1[2] + gt_t[2] * baseline_scale,
    ];

    // Project a 3D point P into camera with centre C and rotation R.
    // coords in camera = R * (P - C), projected = [x/z, y/z].
    let project = |p: &[f32; 3], c: &[f32; 3], r: &[[f32; 3]; 3]| -> [f32; 2] {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let xc = r[0][0] * d[0] + r[0][1] * d[1] + r[0][2] * d[2];
        let yc = r[1][0] * d[0] + r[1][1] * d[1] + r[1][2] * d[2];
        let zc = (r[2][0] * d[0] + r[2][1] * d[1] + r[2][2] * d[2]).max(0.01);
        [xc / zc, yc / zc]
    };

    let pts2d_cam1: Vec<[f32; 2]> = scene_pts.iter().map(|p| project(p, &c1, &r1)).collect();
    let pts2d_cam2: Vec<[f32; 2]> = scene_pts.iter().map(|p| project(p, &c2, &gt_rot)).collect();

    // ── Outlier correspondences (random, added after inliers) ─────────────────
    // Derive sub-seeds from state.seed so randomize scrambles all three.
    let seed1 = state.seed.wrapping_mul(0xDEAD_BEEF_u64).wrapping_add(1);
    let seed2 = state.seed.wrapping_mul(0xCAFE_BABE_u64).wrapping_add(2);
    let outlier_pts2d_1 = gen_outlier_pts2d(n_outlier_pts, seed1);
    let outlier_pts2d_2 = gen_outlier_pts2d(n_outlier_pts, seed2);

    // ── Fundamental matrix from known pose: F = [t]_x R (calibrated, K=I) ────
    // F * x1 = epipolar line l2 in cam2 for point x1 in cam1.
    let f_mat = fundamental_from_pose(&gt_t, &gt_rot);

    // ── Confusion matrix ──────────────────────────────────────────────────────
    // Build combined correspondence list: GT inliers first, then GT outliers.
    let all_p1: Vec<[f32; 2]> = pts2d_cam1.iter().copied()
        .chain(outlier_pts2d_1.iter().copied()).collect();
    let all_p2: Vec<[f32; 2]> = pts2d_cam2.iter().copied()
        .chain(outlier_pts2d_2.iter().copied()).collect();
    let n_total = all_p1.len();

    // est_inlier[i] = true if algorithm classified pair i as an inlier.
    let est_inlier: Vec<bool> = match state.algo {
        RansacAlgo::Simple | RansacAlgo::Msac => {
            // Threshold Sampson error against the user's slider.
            let the = state.sampson_threshold;
            all_p1.iter().zip(all_p2.iter())
                .map(|(&p1, &p2)| sampson_error(&f_mat, p1, p2) < the)
                .collect()
        }
        RansacAlgo::Magsac | RansacAlgo::MagsacPP => {
            // Use the inlier crate's essential matrix estimator.
            use inlier::types::DataMatrix;
            let flat1: Vec<f64> = all_p1.iter().flat_map(|p| [p[0] as f64, p[1] as f64]).collect();
            let flat2: Vec<f64> = all_p2.iter().flat_map(|p| [p[0] as f64, p[1] as f64]).collect();
            let dm1 = DataMatrix::from_row_slice(n_total, 2, &flat1);
            let dm2 = DataMatrix::from_row_slice(n_total, 2, &flat2);
            let settings = state.algo.make_settings(state.sampson_threshold as f64, Some(state.seed));
            match inlier::estimate_essential_matrix(&dm1, &dm2, state.sampson_threshold as f64, Some(settings)) {
                Ok(res) => {
                    let inlier_set: std::collections::HashSet<usize> =
                        res.inliers.into_iter().collect();
                    (0..n_total).map(|i| inlier_set.contains(&i)).collect()
                }
                Err(_) => vec![false; n_total],
            }
        }
    };

    let mut conf_tp = 0usize; let mut conf_fp = 0usize;
    let mut conf_tn = 0usize; let mut conf_fn = 0usize;
    for i in 0..n_total {
        let gt_in = i < n_inlier_pts;
        match (gt_in, est_inlier[i]) {
            (true,  true)  => conf_tp += 1,
            (false, true)  => conf_fp += 1,
            (true,  false) => conf_fn += 1,
            (false, false) => conf_tn += 1,
        }
    }
    state.conf_tp = conf_tp;
    state.conf_fp = conf_fp;
    state.conf_tn = conf_tn;
    state.conf_fn = conf_fn;

    // ── Spawn 3D scene points (gray cubes) ────────────────────────────────────
    if state.show_3d_points {
        let cube_mesh = make_cube_mesh_batch(&scene_pts, 0.05);
        commands.spawn((
            Mesh3d(meshes.add(cube_mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.55, 0.55, 0.55),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            EssentialMatrixEntity,
        ));
    }

    // ── Spawn camera frustums ─────────────────────────────────────────────────
    // Cam1: blue wireframe
    spawn_frustum(
        &mut commands, &mut meshes, &mut materials,
        c1, r1,
        Color::srgb(0.2, 0.5, 1.0),
    );
    // Cam2: orange wireframe
    spawn_frustum(
        &mut commands, &mut meshes, &mut materials,
        c2, gt_rot,
        Color::srgb(1.0, 0.55, 0.1),
    );

    // ── Spawn image-plane projected dots (cam1 = blue, cam2 = orange) ─────────
    spawn_image_plane_dots(
        &mut commands, &mut meshes, &mut materials,
        c1, r1, &pts2d_cam1,
        Color::srgb(0.3, 0.6, 1.0),
    );
    spawn_image_plane_dots(
        &mut commands, &mut meshes, &mut materials,
        c2, gt_rot, &pts2d_cam2,
        Color::srgb(1.0, 0.6, 0.2),
    );

    // ── Baseline: thick white line between camera centres ─────────────────────
    {
        let line_mesh = make_line_mesh(&[c1, c2]);
        commands.spawn((
            Mesh3d(meshes.add(line_mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            EssentialMatrixEntity,
        ));
        // Thicken with a cylinder-like tube
        let tube = make_tube_mesh(c1, c2, 0.012);
        commands.spawn((
            Mesh3d(meshes.add(tube)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            EssentialMatrixEntity,
        ));
    }

    // ── Epipolar lines (first 5 inliers, shown in both cameras) ──────────────
    if state.show_epipolar {
        let epipolar_colors: [[f32; 3]; 5] = [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [1.0, 1.0, 0.2],
            [0.8, 0.2, 1.0],
            [0.2, 1.0, 0.9],
        ];
        let n_epi = pts2d_cam1.len().min(5);
        for i in 0..n_epi {
            let [r, g, b] = epipolar_colors[i % 5];
            let col = Color::srgb(r, g, b);

            // Epipolar line in cam2 for point x1 in cam1:  l2 = F * x1
            let x1 = pts2d_cam1[i];
            let l2 = mat3x1_mul_vec2h(&f_mat, x1);

            // Epipolar line in cam1 for point x2 in cam2:  l1 = F^T * x2
            let x2 = pts2d_cam2[i];
            let l1 = mat3x1_mul_vec2h(&transpose3(&f_mat), x2);

            // Draw epipolar line on cam2 image plane
            if let Some((a, b_pt)) = epipolar_line_endpoints_in_image_plane(l2, 0.4, 0.3) {
                let world_a = image_to_world(a, c2, gt_rot);
                let world_b = image_to_world(b_pt, c2, gt_rot);
                let line = make_line_mesh(&[world_a, world_b]);
                commands.spawn((
                    Mesh3d(meshes.add(line)),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: col,
                        unlit: true,
                        ..default()
                    })),
                    Transform::default(),
                    EssentialMatrixEntity,
                ));
            }

            // Draw epipolar line on cam1 image plane
            if let Some((a, b_pt)) = epipolar_line_endpoints_in_image_plane(l1, 0.4, 0.3) {
                let world_a = image_to_world(a, c1, r1);
                let world_b = image_to_world(b_pt, c1, r1);
                let line = make_line_mesh(&[world_a, world_b]);
                commands.spawn((
                    Mesh3d(meshes.add(line)),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: col,
                        unlit: true,
                        ..default()
                    })),
                    Transform::default(),
                    EssentialMatrixEntity,
                ));
            }

            // Also draw a small dot at the image point in cam1 (to show which point it is)
            let wp1 = image_to_world(x1, c1, r1);
            let dot1 = make_cube_mesh_batch(&[wp1], 0.04);
            commands.spawn((
                Mesh3d(meshes.add(dot1)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: col,
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                EssentialMatrixEntity,
            ));
        }
    }

    // ── Outlier dots (red) ────────────────────────────────────────────────────
    {
        let n_show = outlier_pts2d_1.len().min(n_outlier_pts);
        let out_world_1: Vec<[f32; 3]> = outlier_pts2d_1[..n_show]
            .iter()
            .map(|p| image_to_world(*p, c1, r1))
            .collect();
        let out_world_2: Vec<[f32; 3]> = outlier_pts2d_2[..n_show]
            .iter()
            .map(|p| image_to_world(*p, c2, gt_rot))
            .collect();
        if !out_world_1.is_empty() {
            let m = make_cube_mesh_batch(&out_world_1, 0.025);
            commands.spawn((
                Mesh3d(meshes.add(m)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.1, 0.1),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                EssentialMatrixEntity,
            ));
        }
        if !out_world_2.is_empty() {
            let m = make_cube_mesh_batch(&out_world_2, 0.025);
            commands.spawn((
                Mesh3d(meshes.add(m)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.3, 0.05),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                EssentialMatrixEntity,
            ));
        }
    }
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

/// Sampson error for a correspondence (x1, x2) under fundamental matrix F.
/// This is the first-order approximation to the reprojection error.
fn sampson_error(f: &[[f32; 3]; 3], x1: [f32; 2], x2: [f32; 2]) -> f32 {
    // Fx1
    let fx1 = [
        f[0][0] * x1[0] + f[0][1] * x1[1] + f[0][2],
        f[1][0] * x1[0] + f[1][1] * x1[1] + f[1][2],
        f[2][0] * x1[0] + f[2][1] * x1[1] + f[2][2],
    ];
    // F^T x2
    let ftx2 = [
        f[0][0] * x2[0] + f[1][0] * x2[1] + f[2][0],
        f[0][1] * x2[0] + f[1][1] * x2[1] + f[2][1],
        f[0][2] * x2[0] + f[1][2] * x2[1] + f[2][2],
    ];
    // x2^T F x1
    let num = x2[0] * fx1[0] + x2[1] * fx1[1] + fx1[2];
    let denom = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
    if denom < 1e-10 { return 0.0; }
    (num * num) / denom
}

/// Build rotation matrix from Euler angles (roll=rx, pitch=ry, yaw=rz).
fn euler_to_rot(rx: f32, ry: f32, rz: f32) -> [[f32; 3]; 3] {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    // R = Rz * Ry * Rx
    [
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ]
}

/// Angle of rotation (degrees) from a rotation matrix.
fn rot_to_angle_deg(r: &[[f32; 3]; 3]) -> f32 {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    cos_a.acos().to_degrees()
}

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = len3(v).max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Skew-symmetric matrix [t]_x.
fn skew(t: [f32; 3]) -> [[f32; 3]; 3] {
    [
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ]
}

/// 3x3 matrix multiply.
fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn transpose3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Compute F = [t]_x R for calibrated cameras (K=I).
fn fundamental_from_pose(t: &[f32; 3], r: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let tx = skew(*t);
    mat3_mul(&tx, r)
}

/// Multiply 3x3 matrix by homogeneous 2D point [x, y, 1].
fn mat3x1_mul_vec2h(m: &[[f32; 3]; 3], p: [f32; 2]) -> [f32; 3] {
    let v = [p[0], p[1], 1.0_f32];
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Compute the two endpoints of an epipolar line `l = [a, b, c]` (ax + by + c = 0)
/// clipped to the image bounds [-half_w, half_w] x [-half_h, half_h].
/// Returns None if the line does not cross the image.
fn epipolar_line_endpoints_in_image_plane(
    l: [f32; 3],
    half_w: f32,
    half_h: f32,
) -> Option<([f32; 2], [f32; 2])> {
    let [a, b, c] = l;
    // Avoid degenerate lines.
    if a.abs() < 1e-9 && b.abs() < 1e-9 {
        return None;
    }

    // Collect intersection candidates with the four image edges.
    let mut pts: Vec<[f32; 2]> = Vec::new();

    // x = -half_w → y = -(a * (-half_w) + c) / b
    if b.abs() > 1e-9 {
        let y = -(a * (-half_w) + c) / b;
        if y >= -half_h && y <= half_h {
            pts.push([-half_w, y]);
        }
        // x = half_w
        let y2 = -(a * half_w + c) / b;
        if y2 >= -half_h && y2 <= half_h {
            pts.push([half_w, y2]);
        }
    }
    // y = -half_h → x = -(b * (-half_h) + c) / a
    if a.abs() > 1e-9 {
        let x = -(b * (-half_h) + c) / a;
        if x >= -half_w && x <= half_w {
            pts.push([x, -half_h]);
        }
        // y = half_h
        let x2 = -(b * half_h + c) / a;
        if x2 >= -half_w && x2 <= half_w {
            pts.push([x2, half_h]);
        }
    }

    if pts.len() < 2 {
        return None;
    }
    Some((pts[0], pts[1]))
}

/// Convert a 2D image-plane coordinate [x, y] to a 3D world position
/// on the camera's image plane (at z=1 in camera space), transformed by R and C.
fn image_to_world(p: [f32; 2], cam_center: [f32; 3], rot: [[f32; 3]; 3]) -> [f32; 3] {
    // Image plane at z=1 in camera space: (px, py, 1)
    // World = R^T * cam_pt + C
    let r_t = transpose3(&rot);
    let cp = [p[0], p[1], 1.0_f32];
    let wp = mat_vec3(&r_t, cp);
    let scale = 0.3_f32; // distance to near plane
    [
        cam_center[0] + wp[0] * scale,
        cam_center[1] + wp[1] * scale,
        cam_center[2] + wp[2] * scale,
    ]
}

fn mat_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ── Scene point generation ─────────────────────────────────────────────────────

/// Generate `n` 3D scene points in the box x∈[-1,1], y∈[-0.8,0.8], z∈[3.5,6.5]
/// using an LCG with the given seed.
fn gen_scene_points(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut s = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as f32 / u32::MAX as f32
    };
    (0..n)
        .map(|_| {
            let x = next() * 2.0 - 1.0;
            let y = next() * 1.6 - 0.8;
            let z = next() * 3.0 + 3.5;
            [x, y, z]
        })
        .collect()
}

/// Generate `n` random 2D outlier points in [-0.4, 0.4]^2.
fn gen_outlier_pts2d(n: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut s = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as f32 / u32::MAX as f32
    };
    (0..n)
        .map(|_| {
            let x = next() * 0.8 - 0.4;
            let y = next() * 0.6 - 0.3;
            [x, y]
        })
        .collect()
}

// ── Mesh builders ─────────────────────────────────────────────────────────────

/// Build a batch of small cubes at the given positions.
fn make_cube_mesh_batch(positions: &[[f32; 3]], size: f32) -> Mesh {
    let h = size * 0.5;
    let cube_verts: [[f32; 3]; 8] = [
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
    ];
    const CUBE_TRIS: [[u32; 3]; 12] = [
        [0,1,2],[0,2,3], [4,6,5],[4,7,6],
        [0,5,1],[0,4,5], [2,6,7],[2,7,3],
        [0,3,7],[0,7,4], [1,5,6],[1,6,2],
    ];
    let n = positions.len();
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(n * 8);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 36);
    for (i, p) in positions.iter().enumerate() {
        let base = (i * 8) as u32;
        for v in &cube_verts {
            verts.push([p[0] + v[0], p[1] + v[1], p[2] + v[2]]);
        }
        for tri in &CUBE_TRIS {
            indices.push(base + tri[0]);
            indices.push(base + tri[1]);
            indices.push(base + tri[2]);
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Build a LineList mesh from a slice of points (pairs).
fn make_line_mesh(points: &[[f32; 3]]) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, points.to_vec());
    mesh
}

/// Build a tube (thin rectangular prism) between two 3D points.
fn make_tube_mesh(a: [f32; 3], b: [f32; 3], radius: f32) -> Mesh {
    let dir = normalize3([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    // Pick a perpendicular
    let up = if dir[1].abs() < 0.9 { [0.0_f32, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let right = normalize3(cross3(dir, up));
    let up2 = normalize3(cross3(right, dir));

    let r = radius;
    let corners: [[f32; 3]; 4] = [
        add3(scale3(right, r), scale3(up2, r)),
        add3(scale3(right, -r), scale3(up2, r)),
        add3(scale3(right, -r), scale3(up2, -r)),
        add3(scale3(right, r), scale3(up2, -r)),
    ];

    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // 8 vertices: 4 at each end
    for &c in &corners {
        verts.push(add3(a, c));
    }
    for &c in &corners {
        verts.push(add3(b, c));
    }
    // 4 quad faces (each as 2 tris)
    for i in 0..4u32 {
        let j = (i + 1) % 4;
        let a0 = i;
        let a1 = j;
        let b0 = 4 + i;
        let b1 = 4 + j;
        indices.extend([a0, a1, b0, a1, b1, b0]);
    }
    // End caps
    indices.extend([0, 1, 2, 0, 2, 3]); // near cap
    indices.extend([4, 6, 5, 4, 7, 6]); // far cap

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ── Frustum spawner ───────────────────────────────────────────────────────────

/// Spawn a camera frustum wireframe: 4 rays from centre to near-plane corners,
/// plus the near-plane quad.
///
/// Near plane at distance 0.3 from C, half-extents 0.4 × 0.3.
/// Corner in camera space: (±0.4, ±0.3, 1) × 0.3.
fn spawn_frustum(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    cam_center: [f32; 3],
    rot: [[f32; 3]; 3],
    color: Color,
) {
    let near_dist = 0.3_f32;
    let hw = 0.4_f32;
    let hh = 0.3_f32;

    // Near plane corners in camera space (z=1 direction, scaled by near_dist)
    let corners_cam = [
        [hw, hh, 1.0_f32],
        [-hw, hh, 1.0_f32],
        [-hw, -hh, 1.0_f32],
        [hw, -hh, 1.0_f32],
    ];

    // Transform to world space: W = R^T * cam_pt * near_dist + C
    let r_t = transpose3(&rot);
    let corners_world: Vec<[f32; 3]> = corners_cam
        .iter()
        .map(|&c| {
            let scaled = [c[0] * near_dist, c[1] * near_dist, c[2] * near_dist];
            let w = mat_vec3(&r_t, scaled);
            add3(cam_center, w)
        })
        .collect();

    // 4 lines from cam_center to each corner
    let ray_pts: Vec<[f32; 3]> = corners_world
        .iter()
        .flat_map(|&c| [cam_center, c])
        .collect();

    let ray_mesh = {
        let mut m = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, ray_pts);
        m
    };
    commands.spawn((
        Mesh3d(meshes.add(ray_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        EssentialMatrixEntity,
    ));

    // Near-plane quad (wireframe loop: 4 edges)
    let quad_pts: Vec<[f32; 3]> = vec![
        corners_world[0], corners_world[1],
        corners_world[1], corners_world[2],
        corners_world[2], corners_world[3],
        corners_world[3], corners_world[0],
    ];
    let quad_mesh = {
        let mut m = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, quad_pts);
        m
    };
    commands.spawn((
        Mesh3d(meshes.add(quad_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        EssentialMatrixEntity,
    ));

    // Filled near-plane quad (semi-transparent)
    let [c0, c1, c2, c3] = [
        corners_world[0], corners_world[1], corners_world[2], corners_world[3],
    ];
    let plane_verts = vec![c0, c1, c2, c3];
    let plane_indices = Indices::U32(vec![0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]);
    let mut plane_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    plane_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, plane_verts);
    plane_mesh.insert_indices(plane_indices);

    let base_col = color.to_srgba();
    commands.spawn((
        Mesh3d(meshes.add(plane_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(base_col.red, base_col.green, base_col.blue, 0.15),
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            cull_mode: None,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        EssentialMatrixEntity,
    ));
}

/// Spawn tiny colored dots on the image plane surface for each projected point.
fn spawn_image_plane_dots(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    cam_center: [f32; 3],
    rot: [[f32; 3]; 3],
    pts2d: &[[f32; 2]],
    color: Color,
) {
    if pts2d.is_empty() {
        return;
    }
    // Only show points within the image bounds [-0.4, 0.4] x [-0.3, 0.3]
    let hw = 0.4_f32;
    let hh = 0.3_f32;
    let world_pts: Vec<[f32; 3]> = pts2d
        .iter()
        .filter(|&&[x, y]| x.abs() <= hw && y.abs() <= hh)
        .map(|&p| image_to_world(p, cam_center, rot))
        .collect();

    if world_pts.is_empty() {
        return;
    }
    let dot_mesh = make_cube_mesh_batch(&world_pts, 0.018);
    commands.spawn((
        Mesh3d(meshes.add(dot_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        EssentialMatrixEntity,
    ));
}
