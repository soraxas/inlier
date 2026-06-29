//! Gallery demo: 2D RANSAC line fitting on synthetic data.
//!
//! Generates y = 2x + 1 inliers (with ±0.3 noise) plus scattered outliers,
//! then fits a line with inline RANSAC. Points are laid flat on the XZ plane
//! (2D x→world X, 2D y→world Z, world Y=0) and scaled to fit in ±5 units.
//!
//! UI: left panel exposes threshold, n_inliers, n_outliers sliders, a Re-run
//! button, and stats (inlier count, slope/intercept error vs. ground truth).

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use inlier::types::DataMatrix;

use crate::AppDemo;
use crate::algo_config::{RansacAlgo, algo_combo_ui, randomize_button_ui};
use crate::merge_demo::point_cloud_cube_mesh;

pub struct LineFitPlugin;

impl Plugin for LineFitPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LineFitState>()
            .add_systems(OnEnter(AppDemo::LineFit), on_enter)
            .add_systems(OnExit(AppDemo::LineFit), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                line_fit_ui.run_if(in_state(AppDemo::LineFit)),
            )
            .add_systems(Update, line_fit_scene.run_if(in_state(AppDemo::LineFit)));
    }
}

/// Confusion-matrix counts comparing GT labels to estimated inlier set.
#[derive(Default, Clone, Copy)]
pub struct ConfusionStats {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub r#fn: usize,
}

impl ConfusionStats {
    pub fn precision(self) -> Option<f32> {
        let denom = self.tp + self.fp;
        if denom == 0 { None } else { Some(self.tp as f32 / denom as f32) }
    }
    pub fn recall(self) -> Option<f32> {
        let denom = self.tp + self.r#fn;
        if denom == 0 { None } else { Some(self.tp as f32 / denom as f32) }
    }
}

/// Result of the RANSAC line fit.
#[derive(Default, Clone)]
pub struct LineFitResult {
    /// Slope of fitted line (y = slope*x + intercept).
    pub slope: f32,
    /// Intercept of fitted line.
    pub intercept: f32,
    /// Indices of RANSAC inliers into the full point list.
    pub inlier_indices: Vec<usize>,
    /// Confusion stats vs. ground-truth labels.
    pub confusion: ConfusionStats,
}

#[derive(Resource)]
pub struct LineFitState {
    pub needs_update: bool,
    pub threshold: f32,
    pub n_inliers: usize,
    pub n_outliers: usize,
    pub algorithm: RansacAlgo,
    pub seed: u64,
    pub status: String,
    pub result: Option<LineFitResult>,
    /// All generated 2D points [(x, y), ...].
    pub points: Vec<[f32; 2]>,
    /// Ground-truth slope and intercept.
    pub gt_slope: f32,
    pub gt_intercept: f32,
}

impl Default for LineFitState {
    fn default() -> Self {
        Self {
            needs_update: true,
            threshold: 0.4,
            n_inliers: 60,
            n_outliers: 25,
            algorithm: RansacAlgo::default(),
            seed: 42,
            status: "Ready.".into(),
            result: None,
            points: Vec::new(),
            gt_slope: 2.0,
            gt_intercept: 1.0,
        }
    }
}

#[derive(Component)]
struct LineFitEntity;

fn on_enter(mut state: ResMut<LineFitState>) {
    state.needs_update = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<LineFitEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn line_fit_ui(mut contexts: EguiContexts, mut state: ResMut<LineFitState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("line_fit_panel")
        .default_size(280.0)
        .show(ctx, |ui| {
            ui.heading("Line Fitting (RANSAC)");
            ui.separator();
            ui.label("Synthetic: y = 2x + 1");
            ui.label("Green = TP | Red = FP | Blue = FN | Gray = TN");
            ui.label("Yellow box = fitted line");
            ui.separator();

            if algo_combo_ui(ui, &mut state.algorithm) { state.needs_update = true; }

            ui.separator();

            if ui
                .add(egui::Slider::new(&mut state.threshold, 0.05..=3.0_f32).text("Threshold"))
                .changed()
            {
                state.needs_update = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.n_inliers, 10..=400_usize).text("N inliers"))
                .changed()
            {
                state.needs_update = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.n_outliers, 0..=300_usize).text("N outliers"))
                .changed()
            {
                state.needs_update = true;
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Re-run").clicked() { state.needs_update = true; }
                if randomize_button_ui(ui, &mut state.seed) { state.needs_update = true; }
            });

            if let Some(ref r) = state.result.clone() {
                ui.separator();
                ui.colored_label(egui::Color32::LIGHT_GREEN, "Fit Quality");
                let slope_err = (r.slope - state.gt_slope).abs();
                let intercept_err = (r.intercept - state.gt_intercept).abs();
                let slope_color = if slope_err < 0.1 { egui::Color32::GREEN }
                    else if slope_err < 0.3 { egui::Color32::YELLOW }
                    else { egui::Color32::RED };
                ui.colored_label(slope_color, format!("Slope: {:.3} (err {:.3})", r.slope, slope_err));
                let int_color = if intercept_err < 0.1 { egui::Color32::GREEN }
                    else if intercept_err < 0.3 { egui::Color32::YELLOW }
                    else { egui::Color32::RED };
                ui.colored_label(int_color, format!("Intercept: {:.3} (err {:.3})", r.intercept, intercept_err));
                ui.label(format!("GT: y = {:.1}x + {:.1}", state.gt_slope, state.gt_intercept));

                ui.separator();
                let c = r.confusion;
                ui.colored_label(egui::Color32::LIGHT_BLUE, "Confusion Matrix");
                egui::Grid::new("confusion_grid").num_columns(2).show(ui, |ui| {
                    ui.colored_label(egui::Color32::GREEN,  format!("TP {:3}", c.tp));
                    ui.colored_label(egui::Color32::RED,    format!("FP {:3}", c.fp));
                    ui.end_row();
                    ui.colored_label(egui::Color32::from_rgb(100, 180, 255), format!("FN {:3}", c.r#fn));
                    ui.colored_label(egui::Color32::GRAY,   format!("TN {:3}", c.tn));
                    ui.end_row();
                });
                if let Some(p) = c.precision() {
                    let col = if p > 0.9 { egui::Color32::GREEN } else if p > 0.7 { egui::Color32::YELLOW } else { egui::Color32::RED };
                    ui.colored_label(col, format!("Precision: {:.1}%", p * 100.0));
                }
                if let Some(rec) = c.recall() {
                    let col = if rec > 0.9 { egui::Color32::GREEN } else if rec > 0.7 { egui::Color32::YELLOW } else { egui::Color32::RED };
                    ui.colored_label(col, format!("Recall:    {:.1}%", rec * 100.0));
                }
            }

            ui.separator();
            ui.small(&state.status);
        });
}

fn line_fit_scene(
    mut commands: Commands,
    mut state: ResMut<LineFitState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<LineFitEntity>>,
) {
    if !state.needs_update {
        return;
    }
    state.needs_update = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    // Generate points: first n_inliers are GT inliers, rest are GT outliers.
    let pts = generate_line_points(state.n_inliers, state.n_outliers, state.seed);
    state.points = pts.clone();
    let n_gt_inliers = state.n_inliers;

    // Run line fit — use inlier API for RANSAC/MSAC/MAGSAC/MAGSAC++.
    let result = run_line_fit(&pts, state.threshold, state.algorithm, state.seed);
    state.status = format!(
        "({}) inliers: {} / {}",
        state.algorithm.label(),
        result.inlier_indices.len(),
        pts.len()
    );

    // Compute confusion matrix.
    let est_inlier_set: std::collections::HashSet<usize> =
        result.inlier_indices.iter().copied().collect();
    let mut confusion = ConfusionStats::default();
    for i in 0..pts.len() {
        let gt_in = i < n_gt_inliers;
        let est_in = est_inlier_set.contains(&i);
        match (gt_in, est_in) {
            (true,  true)  => confusion.tp += 1,
            (false, true)  => confusion.fp += 1,
            (true,  false) => confusion.r#fn += 1,
            (false, false) => confusion.tn += 1,
        }
    }

    // Scale to fit in ±5 units.
    let scale = 0.7_f32;

    // Four colored groups: TP=green, FP=red, FN=blue, TN=gray.
    let mut tp_pts: Vec<[f32; 3]> = Vec::new();
    let mut fp_pts: Vec<[f32; 3]> = Vec::new();
    let mut fn_pts: Vec<[f32; 3]> = Vec::new();
    let mut tn_pts: Vec<[f32; 3]> = Vec::new();

    for i in 0..pts.len() {
        let p3 = [pts[i][0] * scale, 0.0, pts[i][1] * scale];
        let gt_in = i < n_gt_inliers;
        let est_in = est_inlier_set.contains(&i);
        match (gt_in, est_in) {
            (true,  true)  => tp_pts.push(p3),
            (false, true)  => fp_pts.push(p3),
            (true,  false) => fn_pts.push(p3),
            (false, false) => tn_pts.push(p3),
        }
    }

    for (group_pts, color) in [
        (tp_pts, Color::srgb(0.1, 0.85, 0.2)),
        (fp_pts, Color::srgb(0.9, 0.15, 0.1)),
        (fn_pts, Color::srgb(0.2, 0.4, 1.0)),
        (tn_pts, Color::srgb(0.45, 0.45, 0.45)),
    ] {
        if group_pts.is_empty() { continue; }
        commands.spawn((
            Mesh3d(meshes.add(point_cloud_cube_mesh(&group_pts, 0.12))),
            MeshMaterial3d(materials.add(StandardMaterial { base_color: color, unlit: true, ..default() })),
            Transform::default(),
            LineFitEntity,
        ));
    }

    // Spawn fitted line as a thin elongated yellow box.
    // The line is y = slope * x + intercept.  We lay it in the XZ plane:
    //   world_x = t * scale * dir_x,  world_z = (slope * t + intercept) * scale * dir_z
    // Parameterize along 2D x in [-4, 4].
    let line_half_len = 4.0_f32;
    let dx = 1.0_f32;
    let dz = result.slope;
    let dir_len = (dx * dx + dz * dz).sqrt();
    let dir = Vec3::new(dx / dir_len, 0.0, dz / dir_len);

    // Midpoint of line segment at x=0: world position is (0, 0, intercept*scale).
    let mid = Vec3::new(0.0, 0.0, result.intercept * scale);

    // Rotation from X axis to dir.
    let rotation = Quat::from_rotation_arc(Vec3::X, dir);

    // Box dimensions: long × thin × thin.
    let box_length = line_half_len * 2.0 * scale;
    let line_mesh = make_box_mesh(box_length, 0.04, 0.04);

    commands.spawn((
        Mesh3d(meshes.add(line_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.9, 0.1),
            unlit: true,
            ..default()
        })),
        Transform {
            translation: mid,
            rotation,
            ..default()
        },
        LineFitEntity,
    ));

    state.result = Some(LineFitResult {
        slope: result.slope,
        intercept: result.intercept,
        inlier_indices: result.inlier_indices,
        confusion,
    });
}

// ─── data generation ─────────────────────────────────────────────────────────

/// Deterministic LCG: returns next float in [0, 1).
fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32) / (u32::MAX as f32 + 1.0)
}

/// Returns float in [lo, hi).
fn lcg_range(s: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg_next(s) * (hi - lo)
}

/// Generate 2D points: `n_inliers` near y=2x+1 (x in [-3,3], noise ±0.3),
/// `n_outliers` scattered in [-4,4]×[-7,8].
fn generate_line_points(n_inliers: usize, n_outliers: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut s: u64 = seed;
    let mut pts = Vec::with_capacity(n_inliers + n_outliers);

    for _ in 0..n_inliers {
        let x = lcg_range(&mut s, -3.0, 3.0);
        let noise = lcg_range(&mut s, -0.3, 0.3);
        let y = 2.0 * x + 1.0 + noise;
        pts.push([x, y]);
    }
    for _ in 0..n_outliers {
        let x = lcg_range(&mut s, -4.0, 4.0);
        let y = lcg_range(&mut s, -7.0, 8.0);
        pts.push([x, y]);
    }
    pts
}

// ─── RANSAC line fit ──────────────────────────────────────────────────────────

/// Fit a 2D line ax + bz + c = 0 (normalized: a²+b²=1) from two points.
/// Returns (a, b, c).
fn line_from_two_points(p1: [f32; 2], p2: [f32; 2]) -> [f32; 3] {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let len = (dx * dx + dy * dy).sqrt().max(1e-9);
    // Normal to direction: (-dy, dx) / len
    let a = -dy / len;
    let b = dx / len;
    let c = -(a * p1[0] + b * p1[1]);
    [a, b, c]
}

/// Signed distance from point to line ax+by+c=0 (a²+b²=1 assumed).
#[inline]
fn point_line_dist(p: [f32; 2], line: [f32; 3]) -> f32 {
    (line[0] * p[0] + line[1] * p[1] + line[2]).abs()
}

/// Fit slope+intercept (y = slope*x + intercept) from a set of inlier points
/// using ordinary least squares.
fn ols_line(pts: &[[f32; 2]]) -> (f32, f32) {
    let n = pts.len() as f32;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let sum_x: f32 = pts.iter().map(|p| p[0]).sum();
    let sum_y: f32 = pts.iter().map(|p| p[1]).sum();
    let sum_xx: f32 = pts.iter().map(|p| p[0] * p[0]).sum();
    let sum_xy: f32 = pts.iter().map(|p| p[0] * p[1]).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-9 {
        // Vertical line — return slope=0, intercept=mean_y.
        return (0.0, sum_y / n);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

/// Unified dispatcher: for RANSAC/MSAC uses the inline implementation;
/// for MAGSAC/MAGSAC++ delegates to the inlier crate API.
fn run_line_fit(pts: &[[f32; 2]], threshold: f32, algo: RansacAlgo, seed: u64) -> LineFitResult {
    use RansacAlgo::*;
    match algo {
        Simple => ransac_line_fit(pts, threshold, 500, seed, false),
        Msac   => ransac_line_fit(pts, threshold, 500, seed, true),
        Magsac | MagsacPP => {
            let n = pts.len();
            if n < 2 { return LineFitResult::default(); }
            // Build Nx2 DataMatrix.
            let flat: Vec<f64> = pts.iter().flat_map(|p| [p[0] as f64, p[1] as f64]).collect();
            let data = DataMatrix::from_row_slice(n, 2, &flat);
            let settings = algo.make_settings(threshold as f64, Some(seed));
            match inlier::estimate_line(&data, threshold as f64, Some(settings)) {
                Ok(res) => {
                    let (slope, intercept) = res.model.to_slope_intercept()
                        .unwrap_or((0.0, 0.0));
                    LineFitResult {
                        slope: slope as f32,
                        intercept: intercept as f32,
                        inlier_indices: res.inliers,
                        confusion: ConfusionStats::default(),
                    }
                }
                Err(_) => LineFitResult::default(),
            }
        }
    }
}

/// RANSAC or MSAC line fit (inline). Returns a partial `LineFitResult` (confusion is filled by caller).
/// MSAC uses truncated-quadratic cost (lower = better) instead of inlier count.
fn ransac_line_fit(pts: &[[f32; 2]], threshold: f32, max_iter: usize, seed: u64, msac: bool) -> LineFitResult {
    let n = pts.len();
    if n < 2 {
        return LineFitResult::default();
    }

    let mut s: u64 = seed.wrapping_add(0xdead_beef);
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_line = [0.0f32; 3];
    // For MSAC: lower cost = better. Use f32::MAX as "no best yet".
    let mut best_msac_cost = f32::MAX;

    for _ in 0..max_iter {
        let i = (lcg_next(&mut s) * n as f32) as usize % n;
        let mut j = (lcg_next(&mut s) * n as f32) as usize % n;
        if j == i { j = (i + 1) % n; }

        let line = line_from_two_points(pts[i], pts[j]);

        if msac {
            // MSAC: cost = Σ min(d², t²).  Lower is better.
            let t2 = threshold * threshold;
            let cost: f32 = (0..n)
                .map(|k| { let d = point_line_dist(pts[k], line); (d * d).min(t2) })
                .sum();
            let inliers: Vec<usize> = (0..n)
                .filter(|&k| point_line_dist(pts[k], line) <= threshold)
                .collect();
            if cost < best_msac_cost {
                best_msac_cost = cost;
                best_inliers = inliers;
                best_line = line;
            }
        } else {
            let inliers: Vec<usize> = (0..n)
                .filter(|&k| point_line_dist(pts[k], line) <= threshold)
                .collect();
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_line = line;
            }
        }
    }

    // Refit with OLS over best inliers.
    let inlier_pts: Vec<[f32; 2]> = best_inliers.iter().map(|&i| pts[i]).collect();
    let (slope, intercept) = if inlier_pts.len() >= 2 {
        ols_line(&inlier_pts)
    } else {
        let b = best_line[1];
        if b.abs() > 1e-6 { (-best_line[0] / b, -best_line[2] / b) } else { (0.0, 0.0) }
    };

    // Re-evaluate inliers against the OLS line.
    let a = slope;
    let b_val = -1.0_f32;
    let c = intercept;
    let len = (a * a + b_val * b_val).sqrt().max(1e-9);
    let ols_params = [a / len, b_val / len, c / len];
    let final_inliers: Vec<usize> = (0..n)
        .filter(|&k| point_line_dist(pts[k], ols_params) <= threshold)
        .collect();

    LineFitResult {
        slope,
        intercept,
        inlier_indices: if final_inliers.len() >= best_inliers.len() { final_inliers } else { best_inliers },
        confusion: ConfusionStats::default(),
    }
}

// ─── mesh helpers ─────────────────────────────────────────────────────────────

/// Build an axis-aligned box mesh centred at origin with given dimensions.
/// The box is elongated along X by `len_x`, and has `size_y` × `size_z` cross-section.
fn make_box_mesh(len_x: f32, size_y: f32, size_z: f32) -> Mesh {
    let hx = len_x * 0.5;
    let hy = size_y * 0.5;
    let hz = size_z * 0.5;

    // 8 corners.
    let positions: Vec<[f32; 3]> = vec![
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ];

    let normals: Vec<[f32; 3]> = vec![
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        [0.0, 0.0,  1.0], [0.0, 0.0,  1.0], [0.0, 0.0,  1.0], [0.0, 0.0,  1.0],
    ];

    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        // -Z face
        0, 2, 1,  0, 3, 2,
        // +Z face
        4, 5, 6,  4, 6, 7,
        // -Y face
        0, 1, 5,  0, 5, 4,
        // +Y face
        3, 6, 2,  3, 7, 6,
        // -X face
        0, 4, 7,  0, 7, 3,
        // +X face
        1, 2, 6,  1, 6, 5,
    ];

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
