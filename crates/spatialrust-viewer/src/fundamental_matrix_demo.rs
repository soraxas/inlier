//! Gallery demo: fundamental matrix estimation from 2D-2D point correspondences.
//!
//! Visualizes the geometry of the fundamental matrix without calling any RANSAC
//! estimator.  The scene contains:
//!
//!  - 40 3D points on a planar scene at z = 5
//!  - Camera 1 at the origin, Camera 2 translated by (1.5, 0, 0)
//!  - Both cameras project with a simple perspective model
//!  - 15 random outlier correspondences mixed in
//!
//! Two "image plane" rectangles are displayed side-by-side in 3D world space
//! (left at x = −4, right at x = +4).  Inlier correspondences (green) obey the
//! epipolar constraint; outliers (red) do not.  For a selected inlier the
//! epipolar line on the other plane is drawn in yellow.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use crate::AppDemo;
use crate::algo_config::{RansacAlgo, algo_combo_ui, randomize_button_ui};

// ─── plugin ──────────────────────────────────────────────────────────────────

pub struct FundamentalMatrixPlugin;

impl Plugin for FundamentalMatrixPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FundamentalMatrixState>()
            .add_systems(OnEnter(AppDemo::FundamentalMatrix), on_enter)
            .add_systems(OnExit(AppDemo::FundamentalMatrix), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                fundamental_matrix_ui.run_if(in_state(AppDemo::FundamentalMatrix)),
            )
            .add_systems(
                Update,
                fundamental_matrix_scene.run_if(in_state(AppDemo::FundamentalMatrix)),
            );
    }
}

// ─── state ───────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct FundamentalMatrixState {
    // Parameters
    pub n_inliers: usize,
    pub n_outliers: usize,
    pub seed: u32,
    pub algo: RansacAlgo,

    // Visibility toggles
    pub show_3d_scene: bool,
    pub show_image_planes: bool,
    pub show_correspondences: bool,
    pub show_epipolar: bool,

    // Computed results (filled in by scene system)
    pub inlier_count: usize,
    pub f_det: f64,          // determinant of the 3×3 F matrix (display only)
    pub selected_inlier: usize, // index into inlier list for epipolar line

    // Trigger rebuild
    pub needs_run: bool,
}

impl Default for FundamentalMatrixState {
    fn default() -> Self {
        Self {
            n_inliers: 40,
            n_outliers: 15,
            seed: 42,
            algo: RansacAlgo::default(),
            show_3d_scene: true,
            show_image_planes: true,
            show_correspondences: true,
            show_epipolar: true,
            inlier_count: 0,
            f_det: 0.0,
            selected_inlier: 0,
            needs_run: true,
        }
    }
}

// ─── marker component ────────────────────────────────────────────────────────

#[derive(Component)]
pub struct FundamentalMatrixEntity;

// ─── lifecycle ───────────────────────────────────────────────────────────────

fn on_enter(mut state: ResMut<FundamentalMatrixState>) {
    state.needs_run = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<FundamentalMatrixEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

// ─── UI ──────────────────────────────────────────────────────────────────────

fn fundamental_matrix_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<FundamentalMatrixState>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("fm_panel")
        .default_size(270.0)
        .show(ctx, |ui| {
            ui.heading("Fundamental Matrix");
            ui.separator();
            ui.small("Epipolar geometry from 2 views");
            ui.separator();

            if algo_combo_ui(ui, &mut state.algo) { state.needs_run = true; }
            ui.separator();

            // Parameters
            let c1 = ui
                .add(egui::Slider::new(&mut state.n_inliers, 5..=300).text("Inliers"))
                .changed();
            let c2 = ui
                .add(egui::Slider::new(&mut state.n_outliers, 0..=200).text("Outliers"))
                .changed();
            if c1 || c2 {
                state.needs_run = true;
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Re-run").clicked() {
                    state.needs_run = true;
                }
                let mut seed64 = state.seed as u64;
                if randomize_button_ui(ui, &mut seed64) {
                    state.seed = seed64 as u32;
                    state.needs_run = true;
                }
            });

            // Visibility checkboxes
            ui.separator();
            let v1 = ui.checkbox(&mut state.show_3d_scene, "Show 3D scene").changed();
            let v2 = ui
                .checkbox(&mut state.show_image_planes, "Show image planes")
                .changed();
            let v3 = ui
                .checkbox(&mut state.show_correspondences, "Show correspondences")
                .changed();
            let v4 = ui
                .checkbox(&mut state.show_epipolar, "Show epipolar line (yellow)")
                .changed();
            if v1 || v2 || v3 || v4 {
                state.needs_run = true;
            }

            // Epipolar line selector
            if state.inlier_count > 0 && state.show_epipolar {
                ui.separator();
                let max = state.inlier_count.saturating_sub(1);
                if ui
                    .add(
                        egui::Slider::new(&mut state.selected_inlier, 0..=max)
                            .text("Epipolar point"),
                    )
                    .changed()
                {
                    state.needs_run = true;
                }
            }

            // Stats
            if state.inlier_count > 0 {
                ui.separator();
                ui.colored_label(egui::Color32::LIGHT_BLUE, "Stats");
                ui.label(format!("Inlier correspondences: {}", state.inlier_count));
                ui.label(format!("Outlier correspondences: {}", state.n_outliers));
                ui.separator();
                ui.label("Fundamental matrix F");
                ui.small("(closed-form from camera geometry)");
                ui.label(format!("det(F) = {:.2e}", state.f_det));
                ui.small("det(F) ≈ 0 for a valid F matrix");
                ui.separator();
                ui.label("Epipolar constraint: x2^T F x1 = 0");
                ui.small("Translation [1.5, 0, 0] → epipolar lines");
                ui.small("are horizontal (same row in both views)");
            }

            ui.separator();
            ui.colored_label(egui::Color32::GREEN, "■ Inlier correspondences");
            ui.colored_label(egui::Color32::RED, "■ Outlier correspondences");
            ui.colored_label(egui::Color32::YELLOW, "■ Epipolar line");
            ui.colored_label(egui::Color32::LIGHT_GRAY, "■ 3D scene points");
            ui.colored_label(
                egui::Color32::from_rgb(100, 200, 255),
                "■ Camera 1 (left)",
            );
            ui.colored_label(
                egui::Color32::from_rgb(255, 180, 50),
                "■ Camera 2 (right)",
            );
        });
}

// ─── scene ───────────────────────────────────────────────────────────────────

fn fundamental_matrix_scene(
    mut commands: Commands,
    mut state: ResMut<FundamentalMatrixState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<FundamentalMatrixEntity>>,
) {
    if !state.needs_run {
        return;
    }
    state.needs_run = false;

    // Despawn old entities.
    for e in &existing {
        commands.entity(e).despawn();
    }

    // ── Generate 3D scene points ──────────────────────────────────────────────
    let scene_pts = gen_scene_points(state.n_inliers, state.seed);

    // ── Project to image planes ───────────────────────────────────────────────
    // Camera 1 at origin, Camera 2 at (1.5, 0, 0).
    // Simple pinhole: (x/z, y/z).  Focal length = 1.
    let cam1_pos = Vec3::new(0.0, 0.0, 0.0);
    let cam2_pos = Vec3::new(1.5, 0.0, 0.0);

    let proj1: Vec<Vec2> = scene_pts
        .iter()
        .map(|&p| {
            let px = p[0] - cam1_pos.x;
            let py = p[1] - cam1_pos.y;
            let pz = p[2] - cam1_pos.z;
            Vec2::new(px / pz, py / pz)
        })
        .collect();

    let proj2: Vec<Vec2> = scene_pts
        .iter()
        .map(|&p| {
            let px = p[0] - cam2_pos.x;
            let py = p[1] - cam2_pos.y;
            let pz = p[2] - cam2_pos.z;
            Vec2::new(px / pz, py / pz)
        })
        .collect();

    // ── Generate outlier correspondences ─────────────────────────────────────
    let outlier_pairs = gen_outlier_pairs(state.n_outliers, state.seed);

    // ── Compute the analytical fundamental matrix ─────────────────────────────
    // For cam1 at origin and cam2 at T = [tx, 0, 0] with identity rotation:
    //   F = [T]× where [T]× is the cross-product matrix of T.
    //   F = [[0,  0, 0],
    //        [0,  0, -tx],
    //        [0,  tx, 0]]
    // This is the essential matrix (K = I here), which is also F when f=1.
    let tx = cam2_pos.x;
    // F (column-major indexing for clarity):
    // Row 0: 0,   0,  0
    // Row 1: 0,   0, -tx
    // Row 2: 0,   tx,  0
    let f_mat = [
        [0.0f64, 0.0, 0.0],
        [0.0,    0.0, -tx as f64],
        [0.0,    tx as f64, 0.0],
    ];
    // det of F: computed analytically (it's always 0 for a valid F matrix)
    let f_det = mat3_det(&f_mat);
    state.f_det = f_det;

    // ── Image plane display geometry ─────────────────────────────────────────
    // Left plane center at world x = -4.0, right plane center at x = +4.0.
    // Points are displayed in the YZ plane (Y = proj_x, Z = proj_y) scaled.
    let scale = 1.2f32; // scale projected coords to world units on the plane
    let left_x = -4.0f32;
    let right_x = 4.0f32;

    // Map projected coordinate to world position on the display plane.
    let to_left = |p: Vec2| -> Vec3 {
        Vec3::new(left_x, p.x * scale, p.y * scale)
    };
    let to_right = |p: Vec2| -> Vec3 {
        Vec3::new(right_x, p.x * scale, p.y * scale)
    };

    let n_inliers = proj1.len();
    state.inlier_count = n_inliers;

    // ── Spawn image plane rectangles ──────────────────────────────────────────
    if state.show_image_planes {
        let plane_half_w = 2.0f32;
        let plane_half_h = 1.5f32;
        // Left plane (cam1)
        commands.spawn((
            Mesh3d(meshes.add(make_rect_frame_mesh(
                Vec3::new(left_x, 0.0, 0.0),
                plane_half_w,
                plane_half_h,
                0.025,
                Axis::X,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(0.4, 0.8, 1.0, 0.9),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));
        // Right plane (cam2)
        commands.spawn((
            Mesh3d(meshes.add(make_rect_frame_mesh(
                Vec3::new(right_x, 0.0, 0.0),
                plane_half_w,
                plane_half_h,
                0.025,
                Axis::X,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(1.0, 0.7, 0.2, 0.9),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));

        // Semi-transparent fill for each plane
        commands.spawn((
            Mesh3d(meshes.add(make_rect_fill_mesh(
                Vec3::new(left_x, 0.0, 0.0),
                plane_half_w,
                plane_half_h,
                Axis::X,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(0.4, 0.8, 1.0, 0.08),
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));
        commands.spawn((
            Mesh3d(meshes.add(make_rect_fill_mesh(
                Vec3::new(right_x, 0.0, 0.0),
                plane_half_w,
                plane_half_h,
                Axis::X,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(1.0, 0.7, 0.2, 0.08),
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));
    }

    // ── Spawn projected points on image planes ────────────────────────────────
    if state.show_image_planes {
        let pt_size = 0.06f32;
        // Left (cam1) inlier points — blue
        let left_pts: Vec<[f32; 3]> = proj1.iter().map(|&p| to_left(p).to_array()).collect();
        if !left_pts.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(cube_cloud_mesh(&left_pts, pt_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.3, 0.7, 1.0),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }
        // Right (cam2) inlier points — orange
        let right_pts: Vec<[f32; 3]> = proj2.iter().map(|&p| to_right(p).to_array()).collect();
        if !right_pts.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(cube_cloud_mesh(&right_pts, pt_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.65, 0.1),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }

        // Outlier points on left plane — dark red
        let left_out: Vec<[f32; 3]> = outlier_pairs
            .iter()
            .map(|&(p, _)| to_left(p).to_array())
            .collect();
        if !left_out.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(cube_cloud_mesh(&left_out, pt_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.8, 0.2, 0.2),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }
        // Outlier points on right plane — dark red
        let right_out: Vec<[f32; 3]> = outlier_pairs
            .iter()
            .map(|&(_, p)| to_right(p).to_array())
            .collect();
        if !right_out.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(cube_cloud_mesh(&right_out, pt_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.8, 0.2, 0.2),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }
    }

    // ── Correspondence lines ──────────────────────────────────────────────────
    if state.show_correspondences {
        // Green inlier lines
        if n_inliers > 0 {
            let mut line_pts: Vec<[f32; 3]> = Vec::with_capacity(n_inliers * 2);
            for i in 0..n_inliers {
                line_pts.push(to_left(proj1[i]).to_array());
                line_pts.push(to_right(proj2[i]).to_array());
            }
            commands.spawn((
                Mesh3d(meshes.add(make_line_segments_mesh(&line_pts))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.1, 0.9, 0.1),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }

        // Red outlier lines
        if !outlier_pairs.is_empty() {
            let mut out_lines: Vec<[f32; 3]> = Vec::with_capacity(outlier_pairs.len() * 2);
            for &(pl, pr) in &outlier_pairs {
                out_lines.push(to_left(pl).to_array());
                out_lines.push(to_right(pr).to_array());
            }
            commands.spawn((
                Mesh3d(meshes.add(make_line_segments_mesh(&out_lines))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.95, 0.15, 0.15),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }
    }

    // ── Epipolar line ─────────────────────────────────────────────────────────
    // For the selected inlier, compute the epipolar line l' = F * x1 on image 2.
    // l' = [a, b, c]  →  ay + bz + c = 0  (our image coords: u→Y, v→Z)
    // We draw the line segment clipped to the display rectangle.
    if state.show_epipolar && n_inliers > 0 {
        let sel = state.selected_inlier.min(n_inliers - 1);
        let x1 = proj1[sel]; // (u, v) in cam1 image

        // Homogeneous point x1h = [u, v, 1]
        let x1h = [x1.x as f64, x1.y as f64, 1.0];
        // l' = F * x1h
        let l = mat3_mul_vec3(&f_mat, &x1h);
        // l = [a, b, c]:  a*u + b*v + c = 0  in image2 coords

        // Draw the epipolar line on the right image plane.
        // Image coords span roughly [-1.5, 1.5] in both u and v (at scale 1.2).
        // Clip to [-1.5, 1.5] in u, solve for v.
        let half = 1.5f32 / scale; // image-space half-extent
        let epipolar_pts = epipolar_line_segment(l, half);

        if let Some((p_a, p_b)) = epipolar_pts {
            let w_a = to_right(p_a);
            let w_b = to_right(p_b);
            // Thicker yellow line
            commands.spawn((
                Mesh3d(meshes.add(make_thick_line_mesh(w_a, w_b, 0.04))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.95, 0.0),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }

        // Highlight the selected inlier point on both planes
        let left_sel = to_left(proj1[sel]);
        let right_sel = to_right(proj2[sel]);
        for pos in [left_sel, right_sel] {
            commands.spawn((
                Mesh3d(meshes.add(make_cube_mesh(pos, 0.12))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.95, 0.0),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }
        // Line from selected point on left to selected point on right (yellow)
        commands.spawn((
            Mesh3d(meshes.add(make_thick_line_mesh(left_sel, right_sel, 0.03))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.95, 0.0),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));
    }

    // ── 3D scene points ───────────────────────────────────────────────────────
    if state.show_3d_scene {
        let gray_pts: Vec<[f32; 3]> = scene_pts.iter().map(|&p| p).collect();
        if !gray_pts.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(cube_cloud_mesh(&gray_pts, 0.07))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.6, 0.6, 0.6),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                FundamentalMatrixEntity,
            ));
        }

        // Camera 1 — cyan box
        commands.spawn((
            Mesh3d(meshes.add(make_cube_mesh(cam1_pos, 0.25))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.85, 1.0),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));
        // Camera 2 — gold box
        commands.spawn((
            Mesh3d(meshes.add(make_cube_mesh(cam2_pos, 0.25))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.75, 0.1),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));

        // Projection rays from cam1 to 3D points (thin gray lines)
        let ray_pts: Vec<[f32; 3]> = gray_pts
            .iter()
            .flat_map(|&p| [cam1_pos.to_array(), p])
            .collect();
        commands.spawn((
            Mesh3d(meshes.add(make_line_segments_mesh(&ray_pts))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(0.5, 0.5, 0.5, 0.3),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            FundamentalMatrixEntity,
        ));
    }
}

// ─── math helpers ────────────────────────────────────────────────────────────

/// Multiply a 3×3 matrix (row-major) by a 3-vector.
fn mat3_mul_vec3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Determinant of a 3×3 matrix.
fn mat3_det(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Clip epipolar line [a, b, c] (a*u + b*v + c = 0) to image half-extent.
///
/// Returns two 2D points that lie on the line and inside `[-half, half]^2`,
/// or `None` if the line doesn't cross the visible region.
fn epipolar_line_segment(l: [f64; 3], half: f32) -> Option<(Vec2, Vec2)> {
    let [a, b, c] = l;
    let h = half as f64;

    // Collect intersection candidates with the 4 edges.
    let mut pts: Vec<Vec2> = Vec::new();

    // u = -half, v = -(a*(-h)+c)/b
    if b.abs() > 1e-9 {
        let v = -(a * (-h) + c) / b;
        if v.abs() <= h + 1e-6 {
            pts.push(Vec2::new(-half, v as f32));
        }
        // u = +half
        let v = -(a * h + c) / b;
        if v.abs() <= h + 1e-6 {
            pts.push(Vec2::new(half, v as f32));
        }
    }
    // v = -half
    if a.abs() > 1e-9 {
        let u = -(b * (-h) + c) / a;
        if u.abs() <= h + 1e-6 {
            pts.push(Vec2::new(u as f32, -half));
        }
        // v = +half
        let u = -(b * h + c) / a;
        if u.abs() <= h + 1e-6 {
            pts.push(Vec2::new(u as f32, half));
        }
    }

    // Deduplicate (nearby points from corner intersections)
    pts.dedup_by(|a, b| a.distance(*b) < 1e-4);

    if pts.len() >= 2 {
        Some((pts[0], pts[pts.len() - 1]))
    } else {
        None
    }
}

/// LCG step for seed updates.
fn lcg_u32(seed: u32) -> u32 {
    let s = (seed as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((s >> 33) as u32 % 9_998) + 1
}

// ─── point generation ────────────────────────────────────────────────────────

/// Generate `n` 3D points on a plane at z = 5, x ∈ [−4, 4], y ∈ [−3, 3].
/// Uses an LCG seeded from `seed`.
fn gen_scene_points(n: usize, seed: u32) -> Vec<[f32; 3]> {
    let mut s: u64 = seed as u64 | 1;
    let mut rng = move || -> f32 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    (0..n)
        .map(|_| {
            let x = rng() * 8.0 - 4.0; // [-4, 4]
            let y = rng() * 6.0 - 3.0; // [-3, 3]
            [x, y, 5.0]
        })
        .collect()
}

/// Generate `n` random outlier correspondence pairs.
/// Both image coordinates are random in [−1.5, 1.5].
fn gen_outlier_pairs(n: usize, seed: u32) -> Vec<(Vec2, Vec2)> {
    let mut s: u64 = seed as u64 ^ 0xDEAD_BEEF;
    let mut rng = move || -> f32 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 3.0 - 1.5
    };
    (0..n)
        .map(|_| {
            let l = Vec2::new(rng(), rng());
            let r = Vec2::new(rng(), rng());
            (l, r)
        })
        .collect()
}

// ─── mesh helpers ────────────────────────────────────────────────────────────

/// Which axis is perpendicular to the display rectangle.
enum Axis {
    X,
}

/// Thin frame (outline) rectangle lying in the YZ plane at a given X position.
fn make_rect_frame_mesh(center: Vec3, half_w: f32, half_h: f32, _t: f32, _axis: Axis) -> Mesh {
    let cx = center.x;
    let cy = center.y;
    let cz = center.z;
    // Four edges as thick quads in the YZ plane.
    // Edge segments: (y1,z1) → (y2,z2), extruded by t in Y or Z.
    let segs: [([f32; 3], [f32; 3]); 4] = [
        // bottom
        (
            [cx, cy - half_w, cz - half_h],
            [cx, cy + half_w, cz - half_h],
        ),
        // top
        (
            [cx, cy - half_w, cz + half_h],
            [cx, cy + half_w, cz + half_h],
        ),
        // left
        (
            [cx, cy - half_w, cz - half_h],
            [cx, cy - half_w, cz + half_h],
        ),
        // right
        (
            [cx, cy + half_w, cz - half_h],
            [cx, cy + half_w, cz + half_h],
        ),
    ];

    let mut line_pts: Vec<[f32; 3]> = Vec::new();
    for (a, b) in &segs {
        line_pts.push(*a);
        line_pts.push(*b);
    }
    make_line_segments_mesh(&line_pts)
}

/// Semi-transparent fill rectangle in the YZ plane at a given X position.
fn make_rect_fill_mesh(center: Vec3, half_w: f32, half_h: f32, _axis: Axis) -> Mesh {
    let cx = center.x;
    let cy = center.y;
    let cz = center.z;
    let verts: Vec<[f32; 3]> = vec![
        [cx, cy - half_w, cz - half_h],
        [cx, cy + half_w, cz - half_h],
        [cx, cy + half_w, cz + half_h],
        [cx, cy - half_w, cz + half_h],
    ];
    let norms: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 4];
    let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3, 0, 3, 2, 0, 2, 1];

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Build a line-segments mesh from pairs of points.
/// Uses a thin elongated quad for each segment to make lines visible in 3D.
/// Each pair (pts[2i], pts[2i+1]) is one segment.
fn make_line_segments_mesh(pts: &[[f32; 3]]) -> Mesh {
    // Render as degenerate triangles (2-vertex topology trick):
    // we use a hairline approach — just store raw positions and let
    // the renderer draw them as lines using TriangleList with degenerate tris.
    // For visibility we thicken each segment into a quad along an offset axis.
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut norms: Vec<[f32; 3]> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();

    let pairs = pts.len() / 2;
    for i in 0..pairs {
        let a = Vec3::from(pts[2 * i]);
        let b = Vec3::from(pts[2 * i + 1]);
        let base = verts.len() as u32;

        // Thin billboard quad: offset by a small amount along an arbitrary axis.
        let dir = (b - a).normalize_or_zero();
        // Pick a perpendicular
        let perp = if dir.dot(Vec3::Y).abs() < 0.9 {
            dir.cross(Vec3::Y).normalize_or_zero()
        } else {
            dir.cross(Vec3::Z).normalize_or_zero()
        };
        let off = perp * 0.012;

        verts.extend([
            (a - off).to_array(),
            (a + off).to_array(),
            (b + off).to_array(),
            (b - off).to_array(),
        ]);
        let n = perp.to_array();
        for _ in 0..4 {
            norms.push(n);
        }
        idx.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
        // Back face
        idx.extend([base, base + 2, base + 1, base + 2, base, base + 3]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_indices(Indices::U32(idx));
    mesh
}

/// Thicker line between two world positions, rendered as a round-ish tube
/// (actually a rectangular prism along the segment direction).
fn make_thick_line_mesh(a: Vec3, b: Vec3, half_w: f32) -> Mesh {
    let dir = (b - a).normalize_or_zero();
    let perp1 = if dir.dot(Vec3::Y).abs() < 0.9 {
        dir.cross(Vec3::Y).normalize_or_zero()
    } else {
        dir.cross(Vec3::Z).normalize_or_zero()
    };
    let perp2 = dir.cross(perp1).normalize_or_zero();

    let p1 = perp1 * half_w;
    let p2 = perp2 * half_w;

    // 8 corners of a box
    let corners: [[f32; 3]; 8] = [
        (a - p1 - p2).to_array(),
        (a + p1 - p2).to_array(),
        (a + p1 + p2).to_array(),
        (a - p1 + p2).to_array(),
        (b - p1 - p2).to_array(),
        (b + p1 - p2).to_array(),
        (b + p1 + p2).to_array(),
        (b - p1 + p2).to_array(),
    ];

    // 6 faces × 4 verts each (duplicated for normals)
    let faces: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // -z face (back)
        [4, 7, 6, 5], // +z face (front)
        [0, 4, 5, 1], // -y face (bottom)
        [2, 6, 7, 3], // +y face (top)
        [0, 3, 7, 4], // -x face (left)
        [1, 5, 6, 2], // +x face (right)
    ];

    let face_normals: [[f32; 3]; 6] = [
        (-p2).normalize_or_zero().to_array(),
        p2.normalize_or_zero().to_array(),
        (-p1).normalize_or_zero().to_array(),
        p1.normalize_or_zero().to_array(),
        (-dir).to_array(),
        dir.to_array(),
    ];

    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(24);
    let mut norms: Vec<[f32; 3]> = Vec::with_capacity(24);
    let mut idx: Vec<u32> = Vec::with_capacity(36);

    for (fi, face) in faces.iter().enumerate() {
        let base = verts.len() as u32;
        for &vi in face {
            verts.push(corners[vi]);
            norms.push(face_normals[fi]);
        }
        idx.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_indices(Indices::U32(idx));
    mesh
}

/// Single cube mesh at a given world position and half-size.
fn make_cube_mesh(center: Vec3, size: f32) -> Mesh {
    cube_cloud_mesh(&[center.to_array()], size)
}

/// Build a point cloud as tiny cubes (same approach as `make_cloud_mesh` in plane_demo).
fn cube_cloud_mesh(positions: &[[f32; 3]], size: f32) -> Mesh {
    let h = size * 0.5;
    let n = positions.len();
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(n * 24);
    let mut norms: Vec<[f32; 3]> = Vec::with_capacity(n * 24);
    let mut idx: Vec<u32> = Vec::with_capacity(n * 36);

    for (i, &[cx, cy, cz]) in positions.iter().enumerate() {
        let base = (i * 24) as u32;
        verts.extend([
            [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h],
            [cx + h, cy + h, cz + h],
            [cx + h, cy - h, cz + h],
        ]);
        for _ in 0..4 {
            norms.push([1.0, 0.0, 0.0]);
        }
        verts.extend([
            [cx - h, cy - h, cz + h],
            [cx - h, cy + h, cz + h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz - h],
        ]);
        for _ in 0..4 {
            norms.push([-1.0, 0.0, 0.0]);
        }
        verts.extend([
            [cx - h, cy + h, cz - h],
            [cx - h, cy + h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx + h, cy + h, cz - h],
        ]);
        for _ in 0..4 {
            norms.push([0.0, 1.0, 0.0]);
        }
        verts.extend([
            [cx - h, cy - h, cz + h],
            [cx - h, cy - h, cz - h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy - h, cz + h],
        ]);
        for _ in 0..4 {
            norms.push([0.0, -1.0, 0.0]);
        }
        verts.extend([
            [cx - h, cy - h, cz + h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx - h, cy + h, cz + h],
        ]);
        for _ in 0..4 {
            norms.push([0.0, 0.0, 1.0]);
        }
        verts.extend([
            [cx + h, cy - h, cz - h],
            [cx - h, cy - h, cz - h],
            [cx - h, cy + h, cz - h],
            [cx + h, cy + h, cz - h],
        ]);
        for _ in 0..4 {
            norms.push([0.0, 0.0, -1.0]);
        }
        for face in 0..6u32 {
            let b = base + face * 4;
            idx.extend([b, b + 1, b + 2, b + 2, b + 3, b]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_indices(Indices::U32(idx));
    mesh
}
