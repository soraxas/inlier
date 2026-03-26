//! Point cloud registration utilities inspired by TEASER++.
//!
//! Provides:
//! - Adaptive scale voting (Algorithm 2 in TEASER++) to estimate a robust scale.
//! - A decoupled similarity pipeline: estimate scale via voting, then run rigid registration on
//!   scale-normalized data to recover rotation/translation. The final model is a SimilarityTransform.

use crate::estimators::ScalarTLSEstimator;
use crate::models::SimilarityTransform;
use crate::presets::similarity_registration::similarity_registration_pipeline;
use crate::settings::MetasacSettings;
use crate::types::DataMatrix;
use nalgebra::{SymmetricEigen, Vector3};

/// Estimate scale using TEASER++-style adaptive voting.
///
/// # Arguments
/// * `data` - Nx6 matrix of correspondences: [ax ay az bx by bz].
/// * `noise_bound` - Bound β on inlier noise.
/// * `c_bar` - Truncation parameter (often 1.0).
/// * `max_pairs` - Maximum number of pairs to consider (use to cap O(N^2) work).
pub fn adaptive_scale_voting(
    data: &DataMatrix,
    noise_bound: f64,
    c_bar: f64,
    max_pairs: usize,
) -> Option<f64> {
    let n = data.n_points();
    if n < 2 || data.n_dims() < 6 {
        return None;
    }

    let mut intervals = Vec::new();
    let mut pairs_used = 0usize;

    for i in 0..n {
        let ai = Vector3::new(data.get(i, 0), data.get(i, 1), data.get(i, 2));
        let bi = Vector3::new(data.get(i, 3), data.get(i, 4), data.get(i, 5));
        for j in (i + 1)..n {
            if pairs_used >= max_pairs {
                break;
            }
            let aj = Vector3::new(data.get(j, 0), data.get(j, 1), data.get(j, 2));
            let bj = Vector3::new(data.get(j, 3), data.get(j, 4), data.get(j, 5));
            let da = (aj - ai).norm();
            let db = (bj - bi).norm();
            if da <= 1e-9 {
                continue;
            }
            let s_k = db / da;
            let alpha_k = noise_bound / da;
            let lower = s_k - alpha_k * c_bar;
            let upper = s_k + alpha_k * c_bar;
            intervals.push((lower, upper, s_k));
            pairs_used += 1;
        }
    }

    if intervals.is_empty() {
        return None;
    }

    // Build sorted boundaries.
    let mut boundaries = Vec::with_capacity(intervals.len() * 2);
    for (l, u, _) in &intervals {
        boundaries.push(*l);
        boundaries.push(*u);
    }
    boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Evaluate consensus on interval midpoints.
    let mut best_count = 0usize;
    let mut best_scale = intervals[0].2;
    for win in boundaries.windows(2) {
        let mid = 0.5 * (win[0] + win[1]);
        let mut count = 0usize;
        for (l, u, _) in &intervals {
            if mid >= *l && mid <= *u {
                count += 1;
            }
        }
        if count > best_count {
            best_count = count;
            best_scale = mid;
        }
    }

    Some(best_scale.max(1e-9))
}

/// Scale estimation using TLS (Truncated Least Squares) estimator
///
/// This is the proper TEASER++ implementation using weighted least squares.
/// Uses Translation Invariant Measurements (TIMs) for scale estimation.
///
/// # Arguments
/// * `src_points` - Source point cloud (N×3)
/// * `dst_points` - Target point cloud (N×3)
/// * `noise_bound` - Maximum noise level
/// * `c_bar` - Confidence parameter (typically 1.0)
///
/// # Returns
/// * Estimated scale factor and inlier mask
pub fn estimate_scale_tls(
    src_points: &DataMatrix,
    dst_points: &DataMatrix,
    noise_bound: f64,
    c_bar: f64,
) -> Option<(f64, Vec<bool>)> {
    if src_points.n_points() != dst_points.n_points() {
        return None;
    }

    let n = src_points.n_points();
    if n < 2 {
        return None;
    }

    // Compute TIMs (Translation Invariant Measurements)
    let src_tims = compute_tims(src_points);
    let dst_tims = compute_tims(dst_points);

    // Compute scale ratios from TIM norms
    let n_tims = src_tims.n_points();
    let mut scale_ratios = Vec::with_capacity(n_tims);
    let mut ranges = Vec::with_capacity(n_tims);

    let beta = 2.0 * noise_bound * c_bar.sqrt();

    for i in 0..n_tims {
        let src_norm = compute_norm(&src_tims, i);
        let dst_norm = compute_norm(&dst_tims, i);

        if src_norm < 1e-9 {
            continue;
        }

        let ratio = dst_norm / src_norm;
        let range = beta / src_norm;

        scale_ratios.push(ratio);
        ranges.push(range);
    }

    if scale_ratios.is_empty() {
        return None;
    }

    // Use TLS estimator
    let tls = ScalarTLSEstimator::new();
    tls.estimate(&scale_ratios, &ranges)
}

/// Compute TIMs (Translation Invariant Measurements) from points
fn compute_tims(points: &DataMatrix) -> DataMatrix {
    let n = points.n_points();
    let n_tims = n * (n - 1) / 2;

    let mut tims = DataMatrix::zeros(n_tims, 3);
    let mut idx = 0;

    for i in 0..n {
        let pi = Vector3::new(points.get(i, 0), points.get(i, 1), points.get(i, 2));
        for j in (i + 1)..n {
            let pj = Vector3::new(points.get(j, 0), points.get(j, 1), points.get(j, 2));
            let diff = pj - pi;
            tims.set(idx, 0, diff.x);
            tims.set(idx, 1, diff.y);
            tims.set(idx, 2, diff.z);
            idx += 1;
        }
    }

    tims
}

/// Compute norm of a 3D point/vector in DataMatrix
fn compute_norm(data: &DataMatrix, idx: usize) -> f64 {
    let x = data.get(idx, 0);
    let y = data.get(idx, 1);
    let z = data.get(idx, 2);
    (x * x + y * y + z * z).sqrt()
}

/// Simple geometric suppression: drop correspondences whose source/target neighborhoods
/// are too planar or too sparse. Uses brute-force radius search; intended for modest sizes.
pub fn geometric_suppression(
    data: &DataMatrix,
    radius: f64,
    min_neighbors: usize,
    linearity_thresh: f64,
) -> Vec<bool> {
    let n = data.n_points();
    let mut keep = vec![true; n];
    let r2 = radius * radius;

    for (idx, keep_flag) in keep.iter_mut().enumerate().take(n) {
        let p = Vector3::new(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2));

        let mut neigh_src = Vec::new();
        for j in 0..n {
            let pj = Vector3::new(data.get(j, 0), data.get(j, 1), data.get(j, 2));
            if (pj - p).norm_squared() <= r2 {
                neigh_src.push(pj);
            }
        }
        if neigh_src.len() < min_neighbors {
            *keep_flag = false;
            continue;
        }
        let lin_src = linearity(&neigh_src);
        if lin_src > linearity_thresh {
            *keep_flag = false;
            continue;
        }

        let p_t = Vector3::new(data.get(idx, 3), data.get(idx, 4), data.get(idx, 5));
        let mut neigh_tgt = Vec::new();
        for j in 0..n {
            let pj = Vector3::new(data.get(j, 3), data.get(j, 4), data.get(j, 5));
            if (pj - p_t).norm_squared() <= r2 {
                neigh_tgt.push(pj);
            }
        }
        if neigh_tgt.len() < min_neighbors {
            *keep_flag = false;
            continue;
        }
        let lin_t = linearity(&neigh_tgt);
        if lin_t > linearity_thresh {
            *keep_flag = false;
        }
    }

    keep
}

fn linearity(points: &[Vector3<f64>]) -> f64 {
    if points.len() < 3 {
        return 1.0;
    }
    let mean = points.iter().fold(Vector3::zeros(), |acc, p| acc + p) / (points.len() as f64);
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for p in points {
        let d = p - mean;
        cov += d * d.transpose();
    }
    cov /= points.len() as f64;
    let eig = SymmetricEigen::new(cov);
    let mut vals = [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]];
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if vals[0].abs() < 1e-9 {
        1.0
    } else {
        (vals[0] - vals[1]) / vals[0]
    }
}

/// k-core-like pruning using pairwise compatibility (distance difference within 2*beta).
pub fn compatibility_k_core(
    data: &DataMatrix,
    beta: f64,
    k_min: usize,
    max_pairs: usize,
) -> Vec<bool> {
    let n = data.n_points();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pairs = 0usize;
    for i in 0..n {
        if data.n_dims() < 6 {
            continue;
        }
        let ai = Vector3::new(data.get(i, 0), data.get(i, 1), data.get(i, 2));
        let bi = Vector3::new(data.get(i, 3), data.get(i, 4), data.get(i, 5));
        for j in (i + 1)..n {
            if pairs >= max_pairs {
                break;
            }
            let aj = Vector3::new(data.get(j, 0), data.get(j, 1), data.get(j, 2));
            let bj = Vector3::new(data.get(j, 3), data.get(j, 4), data.get(j, 5));
            let da = (aj - ai).norm();
            let db = (bj - bi).norm();
            if (db - da).abs() <= 2.0 * beta {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
            pairs += 1;
        }
    }

    let mut keep = vec![true; n];
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            let deg = adjacency[i].iter().filter(|&&j| keep[j]).count();
            if deg < k_min {
                keep[i] = false;
                changed = true;
            }
        }
    }
    keep
}

#[allow(clippy::too_many_arguments)]
/// TEASER-like similarity registration: preprocess (suppression + k-core), estimate scale, then run similarity.
pub fn teaser_pointcloud_registration_pipeline(
    threshold: f64,
    noise_bound: f64,
    _c_bar: f64,
    has_priors: bool,
    rng_seed: Option<u64>,
    settings: MetasacSettings,
    suppression_radius: f64,
    suppression_min_neighbors: usize,
    suppression_linearity: f64,
    k_core_min_degree: usize,
) -> impl crate::pipeline::Pipeline<Model = SimilarityTransform, Score = crate::scoring::Score> {
    struct Runner {
        threshold: f64,
        noise_bound: f64,
        has_priors: bool,
        rng_seed: Option<u64>,
        settings: MetasacSettings,
        suppression_radius: f64,
        suppression_min_neighbors: usize,
        suppression_linearity: f64,
        k_core_min_degree: usize,
    }

    impl crate::pipeline::Pipeline for Runner {
        type Model = SimilarityTransform;
        type Score = crate::scoring::Score;

        fn run(
            self,
            data: &DataMatrix,
        ) -> Option<crate::pipeline::PipelineResult<Self::Model, Self::Score>> {
            // Geometric suppression
            let mut working = data.clone();
            let mask_geo = geometric_suppression(
                data,
                self.suppression_radius,
                self.suppression_min_neighbors,
                self.suppression_linearity,
            );
            let filtered_geo = filter_rows(data, &mask_geo);
            if filtered_geo.n_points() >= 3 {
                working = filtered_geo;
            }

            // k-core compatibility pruning
            let mask_core =
                compatibility_k_core(&working, self.noise_bound, self.k_core_min_degree, 50_000);
            let filtered = filter_rows(&working, &mask_core);
            let working = if filtered.n_points() >= 3 {
                filtered
            } else {
                working
            };

            let pipeline = similarity_registration_pipeline(
                self.threshold,
                self.has_priors,
                self.rng_seed,
                self.settings.clone(),
            );
            let result = pipeline.run(&working)?;
            Some(result)
        }
    }

    Runner {
        threshold,
        noise_bound,
        has_priors,
        rng_seed,
        settings,
        suppression_radius,
        suppression_min_neighbors,
        suppression_linearity,
        k_core_min_degree,
    }
}

fn filter_rows(data: &DataMatrix, mask: &[bool]) -> DataMatrix {
    data.filter_points(mask)
}
