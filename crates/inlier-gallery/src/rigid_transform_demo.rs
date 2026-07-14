//! Gallery demo: rigid transform estimation (rotation + translation) from
//! 3D-3D point correspondences.
//!
//! Generates 48 inlier + 12 outlier correspondences, runs a RANSAC Kabsch
//! solver inline, then visualises:
//!   - Source points    → blue cubes
//!   - Target points    → orange cubes
//!   - Inlier lines     → thin gray boxes connecting src↔dst
//!   - Outlier lines    → thin red boxes
//!   - Estimated-transformed source → green cubes
//!
//! Left panel exposes a threshold slider, Re-run, and result stats.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use spatialrust_inlier::normals::{cross3, matvec, normalize3, power_iter};

use crate::algo_config::{algo_combo_ui, randomize_button_ui, RansacAlgo};

use crate::AppDemo;

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct RigidTransformPlugin;

impl Plugin for RigidTransformPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigidTransformState>()
            .add_systems(OnEnter(AppDemo::RigidTransform), on_enter)
            .add_systems(OnExit(AppDemo::RigidTransform), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                rigid_transform_ui.run_if(in_state(AppDemo::RigidTransform)),
            )
            .add_systems(
                Update,
                rigid_transform_scene.run_if(in_state(AppDemo::RigidTransform)),
            );
    }
}

// ── Component marker ─────────────────────────────────────────────────────────

#[derive(Component)]
pub struct RigidTransformEntity;

// ── State resource ────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct RigidTransformState {
    pub threshold: f32,
    pub n_inliers: usize,
    pub n_outliers: usize,
    pub seed: u64,
    pub algo: RansacAlgo,
    pub needs_run: bool,
    // Results
    pub rotation_error_deg: f32,
    pub translation_error: f32,
    pub inlier_count: usize,
    pub status: String,
    // Confusion matrix (GT inlier = first n_inliers pts)
    pub conf_tp: usize,
    pub conf_fp: usize,
    pub conf_tn: usize,
    pub conf_fn: usize,
    // Cached point data
    pub src_pts: Vec<[f32; 3]>,
    pub dst_pts: Vec<[f32; 3]>,
    pub gt_inlier_mask: Vec<bool>, // true = GT inlier (first n_inliers points)
    pub est_rotation: [[f32; 3]; 3],
    pub est_translation: [f32; 3],
    pub est_inlier_mask: Vec<bool>,
}

impl Default for RigidTransformState {
    fn default() -> Self {
        let seed = 0xA11D_ED55_u64;
        let n_in = 48usize;
        let n_out = 12usize;
        let (src, dst, mask) = generate_correspondences(n_in, n_out, seed);
        Self {
            threshold: 0.08,
            n_inliers: n_in,
            n_outliers: n_out,
            seed,
            algo: RansacAlgo::default(),
            needs_run: true,
            rotation_error_deg: 0.0,
            translation_error: 0.0,
            inlier_count: 0,
            status: "Ready.".into(),
            conf_tp: 0,
            conf_fp: 0,
            conf_tn: 0,
            conf_fn: 0,
            src_pts: src,
            dst_pts: dst,
            gt_inlier_mask: mask,
            est_rotation: identity3(),
            est_translation: [0.0; 3],
            est_inlier_mask: Vec::new(),
        }
    }
}

// ── Ground-truth transform ─────────────────────────────────────────────────

/// Ground-truth rotation: ~0.25 rad around Y-axis.
fn gt_rotation() -> [[f32; 3]; 3] {
    let angle: f32 = 0.25;
    let c = angle.cos();
    let s = angle.sin();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Ground-truth translation.
const GT_TRANSLATION: [f32; 3] = [0.65, -0.3, 0.45];

// ── Data generation ───────────────────────────────────────────────────────────

/// LCG random number generator — returns a float in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32 + 1.0)
    }

    /// Returns a value uniformly in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Normal-ish noise via Box-Muller (approximate via sum of uniforms).
    fn noise(&mut self, std: f32) -> f32 {
        // 6-sum approximation to N(0,1).
        let sum: f32 = (0..6).map(|_| self.range(-1.0, 1.0)).sum::<f32>() / 6.0_f32.sqrt();
        sum * std
    }
}

fn rot_apply(r: [[f32; 3]; 3], p: [f32; 3]) -> [f32; 3] {
    matvec(r, p)
}

fn vec_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec_dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = vec_sub(a, b);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Generate inlier + outlier correspondences with the given counts and seed.
///
/// Returns `(src_pts, dst_pts, gt_inlier_mask)`.
fn generate_correspondences(
    n_inliers: usize,
    n_outliers: usize,
    seed: u64,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<bool>) {
    let mut rng = Lcg(seed);
    let r = gt_rotation();
    let t = GT_TRANSLATION;
    let noise_std = 0.015_f32;

    let mut src = Vec::with_capacity(n_inliers + n_outliers);
    let mut dst = Vec::with_capacity(n_inliers + n_outliers);
    let mut mask = Vec::with_capacity(n_inliers + n_outliers);

    // Inlier correspondences: src spread in a box, dst = R*src + t + small noise.
    for _ in 0..n_inliers {
        let p = [
            rng.range(-1.5, 1.5),
            rng.range(-0.8, 0.8),
            rng.range(-1.5, 1.5),
        ];
        let q_clean = vec_add(rot_apply(r, p), t);
        let q = [
            q_clean[0] + rng.noise(noise_std),
            q_clean[1] + rng.noise(noise_std),
            q_clean[2] + rng.noise(noise_std),
        ];
        src.push(p);
        dst.push(q);
        mask.push(true);
    }

    // Outlier correspondences: random positions with no geometric relation.
    for _ in 0..n_outliers {
        let p = [
            rng.range(-1.5, 1.5),
            rng.range(-0.8, 0.8),
            rng.range(-1.5, 1.5),
        ];
        let q = [
            rng.range(-1.5, 1.5),
            rng.range(-0.8, 0.8),
            rng.range(-1.5, 1.5),
        ];
        src.push(p);
        dst.push(q);
        mask.push(false);
    }

    (src, dst, mask)
}

// ── Kabsch solver ─────────────────────────────────────────────────────────────

fn identity3() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat3_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_det(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Kabsch algorithm via SVD of H = A^T B.
///
/// We approximate SVD of a 3×3 matrix using the power-iteration method already
/// present in `spatialrust_inlier::normals` to extract the dominant singular
/// vectors, then handle the remaining 2×2 block analytically.
///
/// Given a set of matched pairs (subsets of src/dst selected by `indices`),
/// returns `(R, t)` such that `R * src_i + t ≈ dst_i` for inliers.
fn kabsch(
    src: &[[f32; 3]],
    dst: &[[f32; 3]],
    indices: &[usize],
) -> Option<([[f32; 3]; 3], [f32; 3])> {
    let n = indices.len();
    if n < 3 {
        return None;
    }
    let nf = n as f32;

    // Centroids.
    let mut cs = [0f32; 3];
    let mut cd = [0f32; 3];
    for &i in indices {
        cs[0] += src[i][0];
        cs[1] += src[i][1];
        cs[2] += src[i][2];
        cd[0] += dst[i][0];
        cd[1] += dst[i][1];
        cd[2] += dst[i][2];
    }
    cs[0] /= nf;
    cs[1] /= nf;
    cs[2] /= nf;
    cd[0] /= nf;
    cd[1] /= nf;
    cd[2] /= nf;

    // Cross-covariance H = sum(centered_dst_i * centered_src_i^T)
    // i.e. H[r][c] += dst_centered[r] * src_centered[c]
    // Then R = V * U^T  where H = U * S * V^T
    let mut h = [[0f32; 3]; 3];
    for &i in indices {
        let ds = [src[i][0] - cs[0], src[i][1] - cs[1], src[i][2] - cs[2]];
        let dd = [dst[i][0] - cd[0], dst[i][1] - cd[1], dst[i][2] - cd[2]];
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += ds[r] * dd[c];
            }
        }
    }
    // We formed H = A^T B in the standard Kabsch sense where A = centered_src, B = centered_dst.
    // Standard formula: H = A^T B, SVD(H) = U S V^T, R = V U^T.
    // Our h matrix above has h[r][c] = sum(ds[r] * dd[c]) = (A^T B) where A=src, B=dst.
    // So h = A^T B. SVD: h = U S V^T, R = V U^T.

    // Approximate SVD via two rounds of power iteration on H H^T (for U) and H^T H (for V).
    let ht = mat3_transpose(h);
    let hht = mat3_mul(h, ht); // H H^T  — eigenvectors are columns of U
    let hth = mat3_mul(ht, h); // H^T H  — eigenvectors are columns of V

    // First singular vector.
    let u0 = power_iter(hht, [0.4, 0.7, 0.3]);
    let v0 = power_iter(hth, [0.4, 0.7, 0.3]);

    // Second singular vector (orthogonal to first).
    // Deflate: B2 = B - sigma1 * v1 v1^T.
    let s1 = (matvec(hth, v0)[0] * v0[0] + matvec(hth, v0)[1] * v0[1] + matvec(hth, v0)[2] * v0[2])
        .max(0.0)
        .sqrt();

    // Deflated H^T H for second vector.
    let hth2 = deflate3(hth, v0, s1 * s1);
    let v1_raw = power_iter(hth2, [0.3, 0.1, 0.9]);
    // Re-orthogonalize v1 against v0.
    let v1 = gram_schmidt(v0, v1_raw);

    // Third singular vector: V2 = v0 × v1 (complete right-handed basis).
    let v2_raw = cross3(v0, v1);
    let v2 = normalize3(v2_raw);

    // Same deflation for U.
    let s1u =
        (matvec(hht, u0)[0] * u0[0] + matvec(hht, u0)[1] * u0[1] + matvec(hht, u0)[2] * u0[2])
            .max(0.0)
            .sqrt();
    let hht2 = deflate3(hht, u0, s1u * s1u);
    let u1_raw = power_iter(hht2, [0.3, 0.1, 0.9]);
    let u1 = gram_schmidt(u0, u1_raw);
    let u2_raw = cross3(u0, u1);
    let u2 = normalize3(u2_raw);

    // Assemble U = [u0 | u1 | u2]  (column-major as rows of the transpose).
    let u_mat = [
        [u0[0], u1[0], u2[0]],
        [u0[1], u1[1], u2[1]],
        [u0[2], u1[2], u2[2]],
    ];
    let v_mat = [
        [v0[0], v1[0], v2[0]],
        [v0[1], v1[1], v2[1]],
        [v0[2], v1[2], v2[2]],
    ];

    // R = V * U^T
    let ut = mat3_transpose(u_mat);
    let mut r = mat3_mul(v_mat, ut);

    // Handle reflection: if det(R) < 0, flip the last column of V.
    if mat3_det(r) < 0.0 {
        let v_mat_fixed = [
            [v_mat[0][0], v_mat[0][1], -v_mat[0][2]],
            [v_mat[1][0], v_mat[1][1], -v_mat[1][2]],
            [v_mat[2][0], v_mat[2][1], -v_mat[2][2]],
        ];
        r = mat3_mul(v_mat_fixed, ut);
    }

    // t = centroid_dst - R * centroid_src
    let r_cs = matvec(r, cs);
    let t = [cd[0] - r_cs[0], cd[1] - r_cs[1], cd[2] - r_cs[2]];

    Some((r, t))
}

/// Deflate a 3×3 symmetric matrix: M' = M - lambda * v * v^T.
fn deflate3(m: [[f32; 3]; 3], v: [f32; 3], lambda: f32) -> [[f32; 3]; 3] {
    let mut out = m;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] -= lambda * v[i] * v[j];
        }
    }
    out
}

/// Remove the component of `b` along `a` (assumes `a` is unit), then normalise.
fn gram_schmidt(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let out = [b[0] - dot * a[0], b[1] - dot * a[1], b[2] - dot * a[2]];
    normalize3(out)
}

// ── RANSAC loop ───────────────────────────────────────────────────────────────

/// Run RANSAC with Kabsch minimal solver (3 samples), count inliers by
/// symmetric distance `|R*src + t - dst|` < threshold.
/// Dispatcher: inline Kabsch RANSAC for Simple/MSAC, inlier API for MAGSAC/MAGSAC++.
fn run_rigid_fit(
    src: &[[f32; 3]],
    dst: &[[f32; 3]],
    threshold: f32,
    algo: RansacAlgo,
    seed: u64,
) -> ([[f32; 3]; 3], [f32; 3], Vec<bool>) {
    use RansacAlgo::*;
    match algo {
        Simple | Msac => ransac_kabsch(src, dst, threshold, 500, seed),
        Magsac | MagsacPP => {
            let n = src.len();
            if n < 3 {
                return (identity3(), [0.0; 3], vec![false; n]);
            }
            let flat_s: Vec<f64> = src
                .iter()
                .flat_map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect();
            let flat_d: Vec<f64> = dst
                .iter()
                .flat_map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect();
            let dm_s = inlier::types::DataMatrix::from_row_slice(n, 3, &flat_s);
            let dm_d = inlier::types::DataMatrix::from_row_slice(n, 3, &flat_d);
            let settings = algo.make_settings(threshold as f64, Some(seed));
            match inlier::estimate_rigid_transform(&dm_s, &dm_d, threshold as f64, Some(settings)) {
                Ok(res) => {
                    let inlier_set: std::collections::HashSet<usize> =
                        res.inliers.into_iter().collect();
                    let mask: Vec<bool> = (0..n).map(|i| inlier_set.contains(&i)).collect();
                    // Extract rotation and translation as [[f32;3];3] and [f32;3].
                    let rot = res.model.rotation.to_rotation_matrix();
                    let m = rot.matrix();
                    let r = [
                        [m[(0, 0)] as f32, m[(0, 1)] as f32, m[(0, 2)] as f32],
                        [m[(1, 0)] as f32, m[(1, 1)] as f32, m[(1, 2)] as f32],
                        [m[(2, 0)] as f32, m[(2, 1)] as f32, m[(2, 2)] as f32],
                    ];
                    let t = [
                        res.model.translation.x as f32,
                        res.model.translation.y as f32,
                        res.model.translation.z as f32,
                    ];
                    (r, t, mask)
                }
                Err(_) => ransac_kabsch(src, dst, threshold, 500, seed),
            }
        }
    }
}

///
/// Returns `(best_R, best_t, inlier_mask)`.
fn ransac_kabsch(
    src: &[[f32; 3]],
    dst: &[[f32; 3]],
    threshold: f32,
    max_iters: usize,
    seed: u64,
) -> ([[f32; 3]; 3], [f32; 3], Vec<bool>) {
    assert_eq!(src.len(), dst.len());
    let n = src.len();
    let mut rng = Lcg(seed);
    let thresh2 = threshold * threshold;

    let mut best_r = identity3();
    let mut best_t = [0f32; 3];
    let mut best_count = 0usize;

    for _ in 0..max_iters {
        // Sample 3 distinct indices.
        let i0 = (rng.next_f32() * n as f32) as usize % n;
        let mut i1 = (rng.next_f32() * n as f32) as usize % n;
        let mut i2 = (rng.next_f32() * n as f32) as usize % n;
        while i1 == i0 {
            i1 = (rng.next_f32() * n as f32) as usize % n;
        }
        while i2 == i0 || i2 == i1 {
            i2 = (rng.next_f32() * n as f32) as usize % n;
        }

        let Some((r, t)) = kabsch(src, dst, &[i0, i1, i2]) else {
            continue;
        };

        let count = src
            .iter()
            .zip(dst.iter())
            .filter(|(s, d)| {
                let xf = vec_add(rot_apply(r, **s), t);
                vec_dist2(xf, **d) < thresh2
            })
            .count();

        if count > best_count {
            best_count = count;
            best_r = r;
            best_t = t;
        }
    }

    // Refit on all inliers.
    let inlier_indices: Vec<usize> = (0..n)
        .filter(|&i| {
            let xf = vec_add(rot_apply(best_r, src[i]), best_t);
            vec_dist2(xf, dst[i]) < thresh2
        })
        .collect();

    if inlier_indices.len() >= 3 {
        if let Some((r2, t2)) = kabsch(src, dst, &inlier_indices) {
            // Recompute inlier mask with refined model.
            let refined_mask: Vec<bool> = (0..n)
                .map(|i| {
                    let xf = vec_add(rot_apply(r2, src[i]), t2);
                    vec_dist2(xf, dst[i]) < thresh2
                })
                .collect();
            let refined_count = refined_mask.iter().filter(|&&b| b).count();
            if refined_count >= best_count {
                return (r2, t2, refined_mask);
            }
        }
    }

    let mask: Vec<bool> = (0..n)
        .map(|i| {
            let xf = vec_add(rot_apply(best_r, src[i]), best_t);
            vec_dist2(xf, dst[i]) < thresh2
        })
        .collect();

    (best_r, best_t, mask)
}

// ── Error metrics ─────────────────────────────────────────────────────────────

/// Rotation error in degrees between two rotation matrices.
fn rotation_error_deg(r_est: [[f32; 3]; 3], r_gt: [[f32; 3]; 3]) -> f32 {
    // R_err = R_est^T * R_gt
    let r_err = mat3_mul(mat3_transpose(r_est), r_gt);
    // trace(R_err) = 1 + 2 cos(theta)
    let trace = r_err[0][0] + r_err[1][1] + r_err[2][2];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

fn translation_error(t_est: [f32; 3], t_gt: [f32; 3]) -> f32 {
    vec_dist2(t_est, t_gt).sqrt()
}

// ── Lifecycle systems ─────────────────────────────────────────────────────────

fn on_enter(mut state: ResMut<RigidTransformState>) {
    state.needs_run = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<RigidTransformEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

// ── UI ────────────────────────────────────────────────────────────────────────

fn rigid_transform_ui(mut contexts: EguiContexts, mut state: ResMut<RigidTransformState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("rigid_transform_panel")
        .default_size(270.0)
        .show(ctx, |ui| {
            ui.heading("Rigid Transform Estimation");
            ui.separator();
            ui.label("GT: R≈0.25 rad around Y, t=[0.65, -0.3, 0.45]");
            ui.separator();

            if algo_combo_ui(ui, &mut state.algo) {
                state.needs_run = true;
            }
            ui.separator();

            if ui
                .add(egui::Slider::new(&mut state.threshold, 0.01..=0.5_f32).text("Threshold"))
                .changed()
            {
                state.needs_run = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.n_inliers, 5..=300_usize).text("N inliers"))
                .changed()
            {
                state.needs_run = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.n_outliers, 0..=200_usize).text("N outliers"))
                .changed()
            {
                state.needs_run = true;
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Re-run").clicked() {
                    state.needs_run = true;
                }
                if randomize_button_ui(ui, &mut state.seed) {
                    state.needs_run = true;
                }
            });

            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Results");
            ui.label(format!("Inlier count:   {}", state.inlier_count));
            ui.label(format!("Rotation error: {:.2}°", state.rotation_error_deg));
            ui.label(format!("Trans error:    {:.4}", state.translation_error));

            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_BLUE, "Confusion Matrix");
            egui::Grid::new("rt_confusion")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.colored_label(egui::Color32::GREEN, format!("TP {:3}", state.conf_tp));
                    ui.colored_label(egui::Color32::RED, format!("FP {:3}", state.conf_fp));
                    ui.end_row();
                    ui.colored_label(
                        egui::Color32::from_rgb(100, 180, 255),
                        format!("FN {:3}", state.conf_fn),
                    );
                    ui.colored_label(egui::Color32::GRAY, format!("TN {:3}", state.conf_tn));
                    ui.end_row();
                });
            let tp = state.conf_tp;
            let fp = state.conf_fp;
            let fn_ = state.conf_fn;
            if tp + fp > 0 {
                let p = tp as f32 / (tp + fp) as f32;
                let c = if p > 0.9 {
                    egui::Color32::GREEN
                } else if p > 0.7 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::RED
                };
                ui.colored_label(c, format!("Precision: {:.1}%", p * 100.0));
            }
            if tp + fn_ > 0 {
                let r = tp as f32 / (tp + fn_) as f32;
                let c = if r > 0.9 {
                    egui::Color32::GREEN
                } else if r > 0.7 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::RED
                };
                ui.colored_label(c, format!("Recall:    {:.1}%", r * 100.0));
            }

            ui.separator();
            ui.small(&state.status);

            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(100, 150, 255),
                "■ Blue   = source pts",
            );
            ui.colored_label(
                egui::Color32::from_rgb(255, 160, 60),
                "■ Orange = target pts",
            );
            ui.colored_label(
                egui::Color32::from_rgb(80, 220, 80),
                "■ Green  = est-transformed src",
            );
            ui.colored_label(
                egui::Color32::from_rgb(180, 180, 180),
                "─ Gray   = inlier correspondence",
            );
            ui.colored_label(
                egui::Color32::from_rgb(220, 60, 60),
                "─ Red    = outlier correspondence",
            );
        });
}

// ── Scene system ──────────────────────────────────────────────────────────────

fn rigid_transform_scene(
    mut commands: Commands,
    mut state: ResMut<RigidTransformState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<RigidTransformEntity>>,
) {
    if !state.needs_run {
        return;
    }
    state.needs_run = false;

    // Clear old entities.
    for e in &existing {
        commands.entity(e).despawn();
    }

    // Regenerate data with current seed / counts.
    let (src, dst, gt_mask) =
        generate_correspondences(state.n_inliers, state.n_outliers, state.seed);
    state.src_pts = src;
    state.dst_pts = dst;
    state.gt_inlier_mask = gt_mask;

    // Run rigid transform estimation.
    let (r_est, t_est, est_mask) = run_rigid_fit(
        &state.src_pts,
        &state.dst_pts,
        state.threshold,
        state.algo,
        state.seed,
    );

    let inlier_count = est_mask.iter().filter(|&&b| b).count();
    let rot_err = rotation_error_deg(r_est, gt_rotation());
    let tr_err = translation_error(t_est, GT_TRANSLATION);

    // Confusion matrix.
    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..state.gt_inlier_mask.len() {
        match (state.gt_inlier_mask[i], est_mask[i]) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    state.conf_tp = tp;
    state.conf_fp = fp;
    state.conf_tn = tn;
    state.conf_fn = fn_;

    state.est_rotation = r_est;
    state.est_translation = t_est;
    state.est_inlier_mask = est_mask.clone();
    state.inlier_count = inlier_count;
    state.rotation_error_deg = rot_err;
    state.translation_error = tr_err;
    state.status = format!(
        "({}) {inlier_count}/{} inliers | R_err={rot_err:.2}° | t_err={tr_err:.4}",
        state.algo.label(),
        state.src_pts.len()
    );

    let n = state.src_pts.len();
    let point_size = 0.055_f32;

    // ── Source points (blue) ──────────────────────────────────────────────────
    {
        let src_pts: Vec<[f32; 3]> = state.src_pts.clone();
        let mesh = point_cloud_cube_mesh_colored(&src_pts, point_size, [0.25, 0.55, 1.0, 1.0]);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            RigidTransformEntity,
        ));
    }

    // ── Target points (orange) ────────────────────────────────────────────────
    {
        let dst_pts: Vec<[f32; 3]> = state.dst_pts.clone();
        let mesh = point_cloud_cube_mesh_colored(&dst_pts, point_size, [1.0, 0.55, 0.1, 1.0]);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            RigidTransformEntity,
        ));
    }

    // ── Estimated-transformed source (green) ──────────────────────────────────
    {
        let transformed: Vec<[f32; 3]> = state
            .src_pts
            .iter()
            .map(|&s| vec_add(rot_apply(r_est, s), t_est))
            .collect();
        let mesh =
            point_cloud_cube_mesh_colored(&transformed, point_size * 0.8, [0.1, 0.85, 0.25, 1.0]);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            RigidTransformEntity,
        ));
    }

    // ── Correspondence line segments ──────────────────────────────────────────
    // Collect inlier and outlier segment endpoints separately for batching.
    let mut inlier_segs: Vec<([f32; 3], [f32; 3])> = Vec::new();
    let mut outlier_segs: Vec<([f32; 3], [f32; 3])> = Vec::new();

    for i in 0..n {
        let a = state.src_pts[i];
        let b = state.dst_pts[i];
        if est_mask[i] {
            inlier_segs.push((a, b));
        } else {
            outlier_segs.push((a, b));
        }
    }

    // Inlier segments (gray).
    if !inlier_segs.is_empty() {
        let mesh = line_segments_mesh(&inlier_segs, 0.008, [0.7, 0.7, 0.7, 0.85]);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            RigidTransformEntity,
        ));
    }

    // Outlier segments (red).
    if !outlier_segs.is_empty() {
        let mesh = line_segments_mesh(&outlier_segs, 0.008, [0.9, 0.18, 0.18, 0.85]);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            RigidTransformEntity,
        ));
    }
}

// ── Mesh helpers ──────────────────────────────────────────────────────────────

/// Build a batched cube-cloud mesh with a uniform RGBA color per point.
fn point_cloud_cube_mesh_colored(pts: &[[f32; 3]], size: f32, color: [f32; 4]) -> Mesh {
    let h = size * 0.5;
    let cube_verts: [[f32; 3]; 8] = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    const CUBE_TRIS: [[u32; 3]; 12] = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [2, 6, 7],
        [2, 7, 3],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ];

    let n = pts.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 8);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 8);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 36);

    for (i, p) in pts.iter().enumerate() {
        let base = (i * 8) as u32;
        for v in &cube_verts {
            positions.push([p[0] + v[0], p[1] + v[1], p[2] + v[2]]);
            colors.push(color);
        }
        for tri in &CUBE_TRIS {
            indices.push(base + tri[0]);
            indices.push(base + tri[1]);
            indices.push(base + tri[2]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Build a batched mesh of thin box line-segments connecting endpoint pairs.
///
/// Each segment is represented as a thin box (capsule approximation) with its
/// long axis aligned to the segment direction and scaled to the segment length.
///
/// `thickness` is the cross-section half-extent (diameter = 2 * thickness).
fn line_segments_mesh(segs: &[([f32; 3], [f32; 3])], thickness: f32, color: [f32; 4]) -> Mesh {
    // Unit box along the Y-axis.  We will transform each vertex by the per-segment
    // matrix computed in pure Rust.
    let h = thickness * 0.5;
    // Unscaled box: length 1 along Y, thickness along X and Z.
    // After scaling: length = segment length, cross = thickness.
    let box_verts: [[f32; 3]; 8] = [
        [-h, -0.5, -h],
        [h, -0.5, -h],
        [h, 0.5, -h],
        [-h, 0.5, -h],
        [-h, -0.5, h],
        [h, -0.5, h],
        [h, 0.5, h],
        [-h, 0.5, h],
    ];
    const BOX_TRIS: [[u32; 3]; 12] = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [2, 6, 7],
        [2, 7, 3],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ];

    let n = segs.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 8);
    let mut colors_out: Vec<[f32; 4]> = Vec::with_capacity(n * 8);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 36);

    for (si, (a, b)) in segs.iter().enumerate() {
        let base = (si * 8) as u32;

        // Midpoint and direction.
        let mid = [
            (a[0] + b[0]) * 0.5,
            (a[1] + b[1]) * 0.5,
            (a[2] + b[2]) * 0.5,
        ];
        let dir_raw = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len = (dir_raw[0] * dir_raw[0] + dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2])
            .sqrt()
            .max(1e-8);
        let dir = [dir_raw[0] / len, dir_raw[1] / len, dir_raw[2] / len];

        // Build a rotation matrix that maps Y → dir.
        let rot = rotation_y_to_dir(dir);

        for v in &box_verts {
            // Scale Y by segment length.
            let vs = [v[0], v[1] * len, v[2]];
            // Rotate.
            let vr = matvec(rot, vs);
            // Translate to midpoint.
            positions.push([mid[0] + vr[0], mid[1] + vr[1], mid[2] + vr[2]]);
            colors_out.push(color);
        }
        for tri in &BOX_TRIS {
            indices.push(base + tri[0]);
            indices.push(base + tri[1]);
            indices.push(base + tri[2]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors_out);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Compute a rotation matrix that maps the Y-axis [0,1,0] onto `dir` (unit vector).
///
/// Uses Rodrigues' rotation formula.
fn rotation_y_to_dir(dir: [f32; 3]) -> [[f32; 3]; 3] {
    let y = [0f32, 1.0, 0.0];
    let dot = y[0] * dir[0] + y[1] * dir[1] + y[2] * dir[2];

    // If dir ≈ Y, return identity.
    if (dot - 1.0).abs() < 1e-6 {
        return identity3();
    }

    // If dir ≈ -Y, return 180° rotation around X.
    if (dot + 1.0).abs() < 1e-6 {
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }

    // Rotation axis k = Y × dir, normalised.
    let k_raw = cross3(y, dir);
    let k = normalize3(k_raw);
    let theta = dot.clamp(-1.0, 1.0).acos();
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;

    // Rodrigues: R = I cos(θ) + (1-cos(θ)) k k^T + sin(θ) [k]×
    [
        [
            t * k[0] * k[0] + c,
            t * k[0] * k[1] - s * k[2],
            t * k[0] * k[2] + s * k[1],
        ],
        [
            t * k[1] * k[0] + s * k[2],
            t * k[1] * k[1] + c,
            t * k[1] * k[2] - s * k[0],
        ],
        [
            t * k[2] * k[0] - s * k[1],
            t * k[2] * k[1] + s * k[0],
            t * k[2] * k[2] + c,
        ],
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_correspondences_sizes() {
        let (src, dst, mask) = generate_correspondences(48, 12, 0xA11D_ED55);
        assert_eq!(src.len(), 60);
        assert_eq!(dst.len(), 60);
        assert_eq!(mask.len(), 60);
        assert_eq!(mask.iter().filter(|&&b| b).count(), 48);
        assert_eq!(mask.iter().filter(|&&b| !b).count(), 12);
    }

    #[test]
    fn kabsch_identity() {
        // If src == dst and no noise, Kabsch should return identity + zero translation.
        let pts: Vec<[f32; 3]> = (0..8)
            .map(|i| {
                let x = (i % 2) as f32;
                let y = ((i / 2) % 2) as f32;
                let z = (i / 4) as f32;
                [x, y, z]
            })
            .collect();
        let indices: Vec<usize> = (0..pts.len()).collect();
        let (r, t) = kabsch(&pts, &pts, &indices).unwrap();
        // R ≈ I, t ≈ 0.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0f32 } else { 0.0 };
                assert!((r[i][j] - expected).abs() < 0.05, "R[{i}][{j}]={}", r[i][j]);
            }
        }
        for i in 0..3 {
            assert!(t[i].abs() < 0.05, "t[{i}]={}", t[i]);
        }
    }

    #[test]
    fn kabsch_known_rotation() {
        // Ground truth correspondences (no noise) — should recover the transform.
        let (src, dst, mask) = generate_correspondences(48, 12, 0xA11D_ED55);
        let inlier_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();

        // Use only the exact inlier pairs (48 points, no noise) — not exact due to noise.
        // Use raw inlier subset (with noise_std=0.015, error should be small).
        let (r, t) = kabsch(&src, &dst, &inlier_indices).unwrap();
        let rot_err = rotation_error_deg(r, gt_rotation());
        let tr_err = translation_error(t, GT_TRANSLATION);

        assert!(rot_err < 2.0, "rotation error too large: {rot_err:.3}°");
        assert!(tr_err < 0.05, "translation error too large: {tr_err:.5}");
    }

    #[test]
    fn ransac_recovers_transform() {
        let (src, dst, _) = generate_correspondences(48, 12, 0xA11D_ED55);
        let (r, t, mask) = ransac_kabsch(&src, &dst, 0.08, 500, 0xBEEF_CAFE);
        let inlier_count = mask.iter().filter(|&&b| b).count();

        // Should recover at least 40 of the 48 inliers.
        assert!(inlier_count >= 40, "too few inliers found: {inlier_count}");

        let rot_err = rotation_error_deg(r, gt_rotation());
        let tr_err = translation_error(t, GT_TRANSLATION);
        assert!(rot_err < 3.0, "rotation error: {rot_err:.3}°");
        assert!(tr_err < 0.1, "translation error: {tr_err:.5}");
    }

    #[test]
    fn rotation_y_to_dir_maps_correctly() {
        let dir = normalize3([1.0, 1.0, 0.0]);
        let r = rotation_y_to_dir(dir);
        let y = [0f32, 1.0, 0.0];
        let out = matvec(r, y);
        let err = vec_dist2(out, dir).sqrt();
        assert!(err < 1e-5, "rotation_y_to_dir error: {err}");
    }

    #[test]
    fn rotation_error_identity() {
        let r = identity3();
        let err = rotation_error_deg(r, r);
        assert!(err < 1e-4, "identity rotation error: {err}");
    }

    #[test]
    fn mat3_det_identity() {
        assert!((mat3_det(identity3()) - 1.0).abs() < 1e-6);
    }
}
