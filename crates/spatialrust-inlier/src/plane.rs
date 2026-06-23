//! Cloud-level plane fitting using inlier's full RANSAC pipeline.
//!
//! This replaces SpatialRust's basic RANSAC plane segmenter with inlier's
//! MSAC scoring + IRLS local optimization — significantly more robust under
//! high outlier ratios.
//!
//! # Example
//!
//! ```rust,ignore
//! use spatialrust_inlier::plane::estimate_plane_from_cloud;
//!
//! let result = estimate_plane_from_cloud(&cloud, 0.02, None)?;
//! println!("normal {:?}, {} inliers", result.normal, result.inlier_cloud.len());
//! ```

use inlier::{
    MetasacSettings, api::estimate_plane,
    core::{MetaSAC, NoopInlierSelector, RansacTerminationCriterion},
    estimators::PlaneEstimator,
    models::Plane3,
    optimisers::LeastSquaresOptimizer,
    samplers::UniformRandomSampler,
    scoring::SigmaConsensusScoring,
    settings::SamplerType,
};
use inlier::choices::SamplerChoice;
use spatialrust_core::{PointCloud, SpatialError, SpatialResult};

use crate::convert::point_cloud_to_data_matrix;

/// Result of a cloud-level plane fit.
pub struct PlaneResult {
    /// Unit normal of the fitted plane.
    pub normal: [f32; 3],
    /// Plane offset d, so that `normal · p + d ≈ 0` for inlier points.
    pub d: f32,
    /// Point cloud of inlier points (on the plane).
    pub inlier_cloud: PointCloud,
    /// Point cloud of outlier points (off the plane).
    pub outlier_cloud: PointCloud,
    /// Number of RANSAC iterations performed.
    pub iterations: usize,
}

/// Fit a plane to a `PointCloud` using inlier's MSAC + IRLS pipeline.
///
/// # Arguments
/// * `cloud` — Input point cloud (must have xyz positions).
/// * `threshold` — Inlier distance threshold in the same unit as the cloud coordinates.
/// * `settings` — Optional [`MetasacSettings`]; `None` uses defaults.
pub fn estimate_plane_from_cloud(
    cloud: &PointCloud,
    threshold: f64,
    settings: Option<MetasacSettings>,
) -> SpatialResult<PlaneResult> {
    let data = point_cloud_to_data_matrix(cloud)?;

    let result = estimate_plane(&data, threshold, settings)
        .map_err(|e| SpatialError::InvalidArgument(e))?;

    // Split into inlier / outlier clouds
    let n = cloud.len();
    let mut inlier_mask = vec![false; n];
    for &idx in &result.inliers {
        inlier_mask[idx] = true;
    }

    let inlier_cloud = extract_by_mask(cloud, &inlier_mask, true)?;
    let outlier_cloud = extract_by_mask(cloud, &inlier_mask, false)?;

    let normal = result.model.normal;
    Ok(PlaneResult {
        normal: [normal.x as f32, normal.y as f32, normal.z as f32],
        d: result.model.d as f32,
        inlier_cloud,
        outlier_cloud,
        iterations: result.iterations,
    })
}

/// Fit a plane using MAGSAC++ (σ-consensus++) with only a `sigma_max` upper bound.
///
/// Unlike MSAC/RANSAC, MAGSAC++ marginalises over the noise scale — you only
/// need to supply a loose upper bound on the expected noise level (e.g. 10× the
/// rough point spacing). It is robust to miscalibration and works well when the
/// exact threshold is unknown.
///
/// # Arguments
/// * `cloud` — Input point cloud (must have xyz positions).
/// * `sigma_max` — Upper bound on inlier noise level (same units as the cloud).
///   A value 5-20× the point spacing is typical. Pass `None` to auto-estimate
///   from the cloud's mean nearest-neighbour distance.
/// * `settings` — Optional [`MetasacSettings`]; `None` uses defaults.
pub fn estimate_plane_magsac(
    cloud: &PointCloud,
    sigma_max: Option<f64>,
    settings: Option<MetasacSettings>,
) -> SpatialResult<PlaneResult> {
    let data = point_cloud_to_data_matrix(cloud)?;
    let n = data.n_points();
    if n < 3 {
        return Err(SpatialError::InvalidArgument(
            "need at least 3 points to fit a plane".into(),
        ));
    }

    let sigma = match sigma_max {
        Some(s) => s,
        None => {
            // Auto-estimate: sample up to 200 points, compute mean NN distance × 5.
            let stride = (n / 200).max(1);
            let pts: Vec<[f64; 3]> = (0..n)
                .step_by(stride)
                .map(|i| [data.get(i, 0), data.get(i, 1), data.get(i, 2)])
                .collect();
            let k = pts.len();
            let mut sum = 0.0f64;
            for i in 0..k {
                let mut best = f64::MAX;
                for j in 0..k {
                    if i == j { continue; }
                    let d = ((pts[i][0]-pts[j][0]).powi(2)
                           + (pts[i][1]-pts[j][1]).powi(2)
                           + (pts[i][2]-pts[j][2]).powi(2)).sqrt();
                    if d < best { best = d; }
                }
                sum += best;
            }
            ((sum / k as f64) * 5.0).max(1e-4)
        }
    };

    let settings = settings.unwrap_or_default();
    let estimator = PlaneEstimator::new();
    let sampler = SamplerChoice::Uniform(UniformRandomSampler::new(settings.rng_seed));

    // DOF=1 for point-to-plane (scalar residual), k_quantile=3.64 (default, ~99th pct).
    let scoring = SigmaConsensusScoring::new(
        sigma,
        1,
        |data: &inlier::types::DataMatrix, model: &Plane3, idx| {
            model.distance(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2))
        },
    );

    let local_optimizer = Some(LeastSquaresOptimizer::new(PlaneEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(PlaneEstimator::new()));
    let termination = RansacTerminationCriterion { confidence: settings.confidence };

    let mut ransac = MetaSAC::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(NoopInlierSelector),
    );
    ransac.run(&data);

    let model = match &ransac.best_model {
        Some(m) => m.clone(),
        None => return Err(SpatialError::InvalidArgument("MAGSAC++ failed to find a plane".into())),
    };

    let mut inlier_mask = vec![false; n];
    for &idx in &ransac.best_inliers {
        inlier_mask[idx] = true;
    }
    let inlier_cloud = extract_by_mask(cloud, &inlier_mask, true)?;
    let outlier_cloud = extract_by_mask(cloud, &inlier_mask, false)?;

    Ok(PlaneResult {
        normal: [model.normal.x as f32, model.normal.y as f32, model.normal.z as f32],
        d: model.d as f32,
        inlier_cloud,
        outlier_cloud,
        iterations: ransac.iteration,
    })
}

/// Fit a plane using MSAC and return only (unit_normal, d, inlier_count).
///
/// Avoids the empty-cloud bug in `estimate_plane_from_cloud` by never creating
/// inlier/outlier sub-clouds. Intended for pipelines that manage point assignment
/// externally (e.g. region-growing + RANSAC).
pub fn fit_plane_msac(
    pts: &[[f32; 3]],
    threshold: f64,
    settings: Option<MetasacSettings>,
) -> Option<([f32; 3], f32, usize)> {
    use inlier::api::estimate_plane;
    if pts.len() < 3 { return None; }
    let n = pts.len();
    let mut data = Vec::with_capacity(3 * n);
    for p in pts { data.push(p[0] as f64); data.push(p[1] as f64); data.push(p[2] as f64); }
    let dm = inlier::types::DataMatrix::from_row_slice(n, 3, &data);
    let result = estimate_plane(&dm, threshold, settings).ok()?;
    let nv = result.model.normal;
    Some(([nv.x as f32, nv.y as f32, nv.z as f32], result.model.d as f32, result.inliers.len()))
}

/// Fit a plane using MAGSAC++ and return only (unit_normal, d, inlier_count).
///
/// Same as `fit_plane_msac` but uses sigma-consensus scoring — threshold-free,
/// marginalises over noise scale up to `sigma_max`.
pub fn fit_plane_magsac_raw(
    pts: &[[f32; 3]],
    sigma_max: f64,
    settings: Option<MetasacSettings>,
) -> Option<([f32; 3], f32, usize)> {
    if pts.len() < 3 { return None; }
    let n = pts.len();
    let mut raw = Vec::with_capacity(3 * n);
    for p in pts { raw.push(p[0] as f64); raw.push(p[1] as f64); raw.push(p[2] as f64); }
    let data = inlier::types::DataMatrix::from_row_slice(n, 3, &raw);

    let settings = settings.unwrap_or_default();
    let estimator = PlaneEstimator::new();
    let sampler = SamplerChoice::Uniform(UniformRandomSampler::new(settings.rng_seed));
    let scoring = SigmaConsensusScoring::new(
        sigma_max,
        1,
        |data: &inlier::types::DataMatrix, model: &Plane3, idx| {
            model.distance(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2))
        },
    );
    let local_optimizer = Some(LeastSquaresOptimizer::new(PlaneEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(PlaneEstimator::new()));
    let termination = RansacTerminationCriterion { confidence: settings.confidence };
    let mut ransac = MetaSAC::new(
        settings, estimator, sampler, scoring,
        local_optimizer, final_optimizer, termination,
        Some(NoopInlierSelector),
    );
    ransac.run(&data);
    let model = ransac.best_model.as_ref()?;
    let nv = model.normal;
    Some(([nv.x as f32, nv.y as f32, nv.z as f32], model.d as f32, ransac.best_inliers.len()))
}

fn extract_by_mask(
    cloud: &PointCloud,
    mask: &[bool],
    keep_true: bool,
) -> SpatialResult<PointCloud> {
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_core::HasPositions3;

    let (xs, ys, zs) = cloud.positions3()?;
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for (i, &flag) in mask.iter().enumerate() {
        if flag == keep_true {
            builder
                .push_point([xs[i], ys[i], zs[i]])
                .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
        }
    }
    builder.build()
}
