//! High-level Rust API for SupeRANSAC.
//!
//! This module provides user-friendly functions for estimating geometric models
//! similar to the Python API.

use crate::core::SuperRansac;
use crate::core::{LeastSquaresOptimizer, NoopInlierSelector};
use crate::estimators::{
    AbsolutePoseEstimator, EssentialEstimator, FundamentalEstimator, HomographyEstimator,
    LineEstimator, RigidTransformEstimator,
};
use crate::models::{
    AbsolutePose, EssentialMatrix, FundamentalMatrix, Homography, Line, RigidTransform,
};
use crate::samplers::UniformRandomSampler;
use crate::scoring::{RansacInlierCountScoring, Score};
use crate::settings::RansacSettings;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Vector2, Vector3};

/// Result of a RANSAC estimation.
#[derive(Debug, Clone)]
pub struct EstimationResult<M> {
    /// The estimated model.
    pub model: M,
    /// Indices of inlier points.
    pub inliers: Vec<usize>,
    /// Score of the estimated model.
    pub score: Score,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Estimate a homography matrix from 2D point correspondences.
///
/// # Arguments
/// * `points1` - First set of 2D points (Nx2 matrix)
/// * `points2` - Second set of 2D points (Nx2 matrix)
/// * `threshold` - Inlier threshold in pixels
/// * `settings` - Optional RANSAC settings (uses defaults if None)
///
/// # Returns
/// `EstimationResult` containing the homography matrix, inliers, score, and iterations.
pub fn estimate_homography(
    points1: &DMatrix<f64>,
    points2: &DMatrix<f64>,
    threshold: f64,
    settings_opt: Option<RansacSettings>,
) -> Result<EstimationResult<Homography>, String> {
    if points1.nrows() != points2.nrows() {
        return Err("points1 and points2 must have the same number of rows".to_string());
    }
    if points1.ncols() != 2 || points2.ncols() != 2 {
        return Err("points must be Nx2 matrices".to_string());
    }

    // Combine into data matrix: [x1, y1, x2, y2]
    let n = points1.nrows();
    let mut data = DataMatrix::zeros(n, 4);
    for i in 0..n {
        data[(i, 0)] = points1[(i, 0)];
        data[(i, 1)] = points1[(i, 1)];
        data[(i, 2)] = points2[(i, 0)];
        data[(i, 3)] = points2[(i, 1)];
    }

    let settings = settings_opt.unwrap_or_default();
    let estimator = HomographyEstimator::new();
    let sampler = UniformRandomSampler::new();
    let scoring_builder =
        RansacInlierCountScoring::new(threshold, |data, model: &Homography, idx| {
            // Compute symmetric transfer error
            let p1 = Vector2::new(data[(idx, 0)], data[(idx, 1)]);
            let p2 = Vector2::new(data[(idx, 2)], data[(idx, 3)]);
            let p1_home = Vector3::new(p1.x, p1.y, 1.0);
            let p2_home = Vector3::new(p2.x, p2.y, 1.0);

            let p2_pred = model.h * p1_home;
            let p1_pred = model.h.transpose() * p2_home;

            let err1 = (p2 - Vector2::new(p2_pred.x / p2_pred.z, p2_pred.y / p2_pred.z)).norm();
            let err2 = (p1 - Vector2::new(p1_pred.x / p1_pred.z, p1_pred.y / p1_pred.z)).norm();

            (err1 + err2) / 2.0
        });
    let scoring = match settings.point_priors.as_ref() {
        Some(priors) if priors.len() == n => scoring_builder.with_priors(priors),
        _ => scoring_builder,
    };
    let local_optimizer = Some(LeastSquaresOptimizer::new(HomographyEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(HomographyEstimator::new()));
    let termination = crate::core::RansacTerminationCriterion { confidence: 0.99 };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(inlier_selector),
    );

    ransac.run(&data);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate homography".to_string()),
    }
}

/// Estimate a fundamental matrix from 2D point correspondences.
///
/// # Arguments
/// * `points1` - First set of 2D points (Nx2 matrix)
/// * `points2` - Second set of 2D points (Nx2 matrix)
/// * `threshold` - Inlier threshold in pixels
/// * `settings` - Optional RANSAC settings (uses defaults if None)
///
/// # Returns
/// `EstimationResult` containing the fundamental matrix, inliers, score, and iterations.
pub fn estimate_fundamental_matrix(
    points1: &DMatrix<f64>,
    points2: &DMatrix<f64>,
    threshold: f64,
    settings_opt: Option<RansacSettings>,
) -> Result<EstimationResult<FundamentalMatrix>, String> {
    if points1.nrows() != points2.nrows() {
        return Err("points1 and points2 must have the same number of rows".to_string());
    }
    if points1.ncols() != 2 || points2.ncols() != 2 {
        return Err("points must be Nx2 matrices".to_string());
    }

    // Combine into data matrix: [x1, y1, x2, y2]
    let n = points1.nrows();
    let mut data = DataMatrix::zeros(n, 4);
    for i in 0..n {
        data[(i, 0)] = points1[(i, 0)];
        data[(i, 1)] = points1[(i, 1)];
        data[(i, 2)] = points2[(i, 0)];
        data[(i, 3)] = points2[(i, 1)];
    }

    let estimator = FundamentalEstimator::new();
    let sampler = UniformRandomSampler::new();
    let scoring_builder =
        RansacInlierCountScoring::new(threshold, |data, model: &FundamentalMatrix, idx| {
            // Compute Sampson error
            let p1 = Vector2::new(data[(idx, 0)], data[(idx, 1)]);
            let p2 = Vector2::new(data[(idx, 2)], data[(idx, 3)]);
            crate::bundle_adjustment::sampson_error(&model.f, &p1, &p2)
        });
    let settings = settings_opt.unwrap_or_default();
    let scoring = match settings.point_priors.as_ref() {
        Some(priors) if priors.len() == n => scoring_builder.with_priors(priors),
        _ => scoring_builder,
    };
    let local_optimizer = Some(LeastSquaresOptimizer::new(FundamentalEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(FundamentalEstimator::new()));
    let termination = crate::core::RansacTerminationCriterion { confidence: 0.99 };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(inlier_selector),
    );

    ransac.run(&data);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate fundamental matrix".to_string()),
    }
}

/// Estimate an essential matrix from 2D point correspondences.
///
/// # Arguments
/// * `points1` - First set of 2D points (Nx2 matrix)
/// * `points2` - Second set of 2D points (Nx2 matrix)
/// * `threshold` - Inlier threshold in pixels
/// * `settings` - Optional RANSAC settings (uses defaults if None)
///
/// # Returns
/// `EstimationResult` containing the essential matrix, inliers, score, and iterations.
pub fn estimate_essential_matrix(
    points1: &DMatrix<f64>,
    points2: &DMatrix<f64>,
    threshold: f64,
    settings_opt: Option<RansacSettings>,
) -> Result<EstimationResult<EssentialMatrix>, String> {
    if points1.nrows() != points2.nrows() {
        return Err("points1 and points2 must have the same number of rows".to_string());
    }
    if points1.ncols() != 2 || points2.ncols() != 2 {
        return Err("points must be Nx2 matrices".to_string());
    }

    // Combine into data matrix: [x1, y1, x2, y2]
    let n = points1.nrows();
    let mut data = DataMatrix::zeros(n, 4);
    for i in 0..n {
        data[(i, 0)] = points1[(i, 0)];
        data[(i, 1)] = points1[(i, 1)];
        data[(i, 2)] = points2[(i, 0)];
        data[(i, 3)] = points2[(i, 1)];
    }

    let settings = settings_opt.unwrap_or_default();
    let estimator = EssentialEstimator::new();
    let sampler = UniformRandomSampler::new();
    let scoring_builder =
        RansacInlierCountScoring::new(threshold, |data, model: &EssentialMatrix, idx| {
            // Compute Sampson error
            let p1 = Vector2::new(data[(idx, 0)], data[(idx, 1)]);
            let p2 = Vector2::new(data[(idx, 2)], data[(idx, 3)]);
            crate::bundle_adjustment::sampson_error(&model.e, &p1, &p2)
        });
    let scoring = match settings.point_priors.as_ref() {
        Some(priors) if priors.len() == n => scoring_builder.with_priors(priors),
        _ => scoring_builder,
    };
    let local_optimizer = Some(LeastSquaresOptimizer::new(EssentialEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(EssentialEstimator::new()));
    let termination = crate::core::RansacTerminationCriterion { confidence: 0.99 };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(inlier_selector),
    );

    ransac.run(&data);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate essential matrix".to_string()),
    }
}

/// Estimate absolute pose (OnP) from 3D-2D point correspondences.
///
/// # Arguments
/// * `points_3d` - 3D points (Nx3 matrix)
/// * `points_2d` - 2D image points (Nx2 matrix)
/// * `threshold` - Inlier threshold in pixels
/// * `settings` - Optional RANSAC settings (uses defaults if None)
///
/// # Returns
/// `EstimationResult` containing the absolute pose (rotation + translation), inliers, score, and iterations.
pub fn estimate_absolute_pose(
    points_3d: &DMatrix<f64>,
    points_2d: &DMatrix<f64>,
    threshold: f64,
    settings_opt: Option<RansacSettings>,
) -> Result<EstimationResult<AbsolutePose>, String> {
    if points_3d.nrows() != points_2d.nrows() {
        return Err("points_3d and points_2d must have the same number of rows".to_string());
    }
    if points_3d.ncols() != 3 || points_2d.ncols() != 2 {
        return Err("points_3d must be Nx3 and points_2d must be Nx2".to_string());
    }

    // Combine into data matrix: [x2d, y2d, x3d, y3d, z3d]
    let n = points_3d.nrows();
    let mut data = DataMatrix::zeros(n, 5);
    for i in 0..n {
        data[(i, 0)] = points_2d[(i, 0)];
        data[(i, 1)] = points_2d[(i, 1)];
        data[(i, 2)] = points_3d[(i, 0)];
        data[(i, 3)] = points_3d[(i, 1)];
        data[(i, 4)] = points_3d[(i, 2)];
    }

    let settings = settings_opt.unwrap_or_default();
    let estimator = AbsolutePoseEstimator::new();
    let sampler = UniformRandomSampler::new();
    let scoring_builder =
        RansacInlierCountScoring::new(threshold, |data, model: &AbsolutePose, idx| {
            // Compute reprojection error
            let p_2d = Vector2::new(data[(idx, 0)], data[(idx, 1)]);
            let p_3d = Vector3::new(data[(idx, 2)], data[(idx, 3)], data[(idx, 4)]);
            crate::bundle_adjustment::reprojection_error(
                model.rotation.to_rotation_matrix().matrix(),
                &model.translation.vector,
                &p_2d,
                &p_3d,
            )
        });
    let scoring = match settings.point_priors.as_ref() {
        Some(priors) if priors.len() == n => scoring_builder.with_priors(priors),
        _ => scoring_builder,
    };
    let local_optimizer = Some(LeastSquaresOptimizer::new(AbsolutePoseEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(AbsolutePoseEstimator::new()));
    let termination = crate::core::RansacTerminationCriterion { confidence: 0.99 };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(inlier_selector),
    );

    ransac.run(&data);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate absolute pose".to_string()),
    }
}

/// Estimate rigid transform from 3D-3D point correspondences.
///
/// # Arguments
/// * `points1` - First set of 3D points (Nx3 matrix)
/// * `points2` - Second set of 3D points (Nx3 matrix)
/// * `threshold` - Inlier threshold
/// * `settings` - Optional RANSAC settings (uses defaults if None)
///
/// # Returns
/// `EstimationResult` containing the rigid transform (rotation + translation), inliers, score, and iterations.
pub fn estimate_rigid_transform(
    points1: &DMatrix<f64>,
    points2: &DMatrix<f64>,
    threshold: f64,
    settings_opt: Option<RansacSettings>,
) -> Result<EstimationResult<RigidTransform>, String> {
    if points1.nrows() != points2.nrows() {
        return Err("points1 and points2 must have the same number of rows".to_string());
    }
    if points1.ncols() != 3 || points2.ncols() != 3 {
        return Err("points must be Nx3 matrices".to_string());
    }

    // Combine into data matrix: [x1, y1, z1, x2, y2, z2]
    let n = points1.nrows();
    let mut data = DataMatrix::zeros(n, 6);
    for i in 0..n {
        data[(i, 0)] = points1[(i, 0)];
        data[(i, 1)] = points1[(i, 1)];
        data[(i, 2)] = points1[(i, 2)];
        data[(i, 3)] = points2[(i, 0)];
        data[(i, 4)] = points2[(i, 1)];
        data[(i, 5)] = points2[(i, 2)];
    }

    let settings = settings_opt.unwrap_or_default();
    let estimator = RigidTransformEstimator::new();
    let sampler = UniformRandomSampler::new();
    let scoring_builder =
        RansacInlierCountScoring::new(threshold, |data, model: &RigidTransform, idx| {
            // Compute point-to-point distance
            let p1 = nalgebra::Point3::new(data[(idx, 0)], data[(idx, 1)], data[(idx, 2)]);
            let p2 = Vector3::new(data[(idx, 3)], data[(idx, 4)], data[(idx, 5)]);
            let p1_rotated = model.rotation.transform_point(&p1);
            let p1_final_vec = p1_rotated.coords + model.translation.vector;
            (p2 - p1_final_vec).norm()
        });
    let scoring = match settings.point_priors.as_ref() {
        Some(priors) if priors.len() == n => scoring_builder.with_priors(priors),
        _ => scoring_builder,
    };
    let local_optimizer = Some(LeastSquaresOptimizer::new(RigidTransformEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(RigidTransformEstimator::new()));
    let termination = crate::core::RansacTerminationCriterion { confidence: 0.99 };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(inlier_selector),
    );

    ransac.run(&data);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate rigid transform".to_string()),
    }
}

/// Estimate a line from 2D points using RANSAC.
///
/// # Arguments
/// * `points` - 2D points (Nx2 matrix, each row is [x, y])
/// * `threshold` - Inlier threshold (distance from point to line)
/// * `settings` - Optional RANSAC settings (uses defaults if None)
///
/// # Returns
/// `EstimationResult` containing the line (ax + by + c = 0), inliers, score, and iterations.
///
/// # Example
/// ```
/// use inlier::api::estimate_line;
/// use inlier::settings::RansacSettings;
/// use nalgebra::DMatrix;
///
/// // Generate some 2D points
/// let mut points = DMatrix::<f64>::zeros(10, 2);
/// points[(0, 0)] = 0.0; points[(0, 1)] = 0.0;
/// points[(1, 0)] = 1.0; points[(1, 1)] = 1.0;
/// // ... add more points
///
/// let threshold = 0.5; // Distance threshold
/// let settings = RansacSettings::default();
///
/// match estimate_line(&points, threshold, Some(settings)) {
///     Ok(result) => {
///         println!("Estimated line: {:?}", result.model.params());
///         println!("Inliers: {}", result.inliers.len());
///     }
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub fn estimate_line(
    points: &DMatrix<f64>,
    threshold: f64,
    settings_opt: Option<RansacSettings>,
) -> Result<EstimationResult<Line>, String> {
    if points.ncols() != 2 {
        return Err("points must be Nx2 matrix (each row is [x, y])".to_string());
    }

    // Use points directly as data matrix
    let n = points.nrows();
    let mut data = DataMatrix::zeros(n, 2);
    for i in 0..n {
        data[(i, 0)] = points[(i, 0)];
        data[(i, 1)] = points[(i, 1)];
    }

    let settings = settings_opt.unwrap_or_default();
    let estimator = LineEstimator::new();
    let sampler = UniformRandomSampler::new();
    let scoring_builder = RansacInlierCountScoring::new(threshold, |data, model: &Line, idx| {
        // Compute distance from point to line
        model.distance_to_point(data[(idx, 0)], data[(idx, 1)])
    });
    let scoring = match settings.point_priors.as_ref() {
        Some(priors) if priors.len() == n => scoring_builder.with_priors(priors),
        _ => scoring_builder,
    };
    let local_optimizer = Some(LeastSquaresOptimizer::new(LineEstimator::new()));
    let final_optimizer = Some(LeastSquaresOptimizer::new(LineEstimator::new()));
    let termination = crate::core::RansacTerminationCriterion { confidence: 0.99 };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        final_optimizer,
        termination,
        Some(inlier_selector),
    );

    ransac.run(&data);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate line".to_string()),
    }
}
