//! High-level Rust API for SupeRANSAC.
//!
//! This module provides user-friendly functions for estimating geometric models
//! similar to the Python API.

use crate::choices::{LocalOptimizerChoice, SamplerChoice};
use crate::core::{Estimator, MetaSAC, NoopInlierSelector, Scoring};
use crate::estimators::{
    AbsolutePoseEstimator, EssentialEstimator, FundamentalEstimator, HomographyEstimator,
    LineEstimator, PlaneEstimator, RigidTransformEstimator,
};
use crate::models::{
    AbsolutePose, EssentialMatrix, FundamentalMatrix, Homography, Line, Plane3, RigidTransform,
};
use crate::optimisers::LeastSquaresOptimizer;
use crate::samplers::{ProsacSampler, UniformRandomSampler};
use crate::scoring::{
    MagsacScoring, MsacScoring, RansacInlierCountScoring, Score, SigmaConsensusScoring,
};
use crate::settings::{LocalOptimizationType, MetasacSettings, SamplerType, ScoringType};
use crate::types::DataMatrix;
use nalgebra::{Vector2, Vector3};

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

type Residual<M> = fn(&DataMatrix, &M, usize) -> f64;

fn validate_threshold(threshold: f64) -> Result<(), String> {
    if threshold.is_finite() && threshold > 0.0 {
        Ok(())
    } else {
        Err("threshold must be finite and greater than zero".to_string())
    }
}

fn validate_finite_matrix(points: &DataMatrix, name: &str) -> Result<(), String> {
    for row in 0..points.n_points() {
        for column in 0..points.n_dims() {
            if !points.get(row, column).is_finite() {
                return Err(format!("{name} must contain only finite coordinates"));
            }
        }
    }
    Ok(())
}

fn has_non_collinear_2d_points(points: &DataMatrix) -> bool {
    if points.n_points() < 3 || points.n_dims() < 2 {
        return false;
    }
    let origin = Vector2::new(points.get(0, 0), points.get(0, 1));
    let Some((_, direction, scale_squared)) = (1..points.n_points())
        .map(|index| {
            let point = Vector2::new(points.get(index, 0), points.get(index, 1));
            let direction = point - origin;
            (index, direction, direction.norm_squared())
        })
        .max_by(|left, right| left.2.total_cmp(&right.2))
    else {
        return false;
    };
    if !scale_squared.is_finite() || scale_squared < 1e-24 {
        return false;
    }
    (1..points.n_points()).any(|index| {
        let point = Vector2::new(points.get(index, 0), points.get(index, 1));
        let offset = point - origin;
        (direction.x * offset.y - direction.y * offset.x).abs() > 1e-10 * scale_squared
    })
}

fn validate_settings(settings: &MetasacSettings, point_count: usize) -> Result<(), String> {
    if settings.min_iterations > settings.max_iterations {
        return Err("min_iterations must not exceed max_iterations".to_string());
    }
    if settings.max_sampling_attempts == 0 {
        return Err("max_sampling_attempts must be greater than zero".to_string());
    }
    if !settings.confidence.is_finite() || !(0.0..=1.0).contains(&settings.confidence) {
        return Err("confidence must be finite and in the range [0, 1]".to_string());
    }
    if let Some(priors) = &settings.point_priors {
        if priors.len() != point_count {
            return Err("point_priors must have one value per input point".to_string());
        }
        if priors
            .iter()
            .any(|prior| !prior.is_finite() || *prior < 0.0)
        {
            return Err("point_priors must contain finite, non-negative values".to_string());
        }
    }
    Ok(())
}

/// Runtime scoring selection for the high-level estimation APIs.
enum ApiScoring<M> {
    Ransac(RansacInlierCountScoring<M, Residual<M>>),
    Msac(MsacScoring<M, Residual<M>>),
    Magsac(MagsacScoring<M, Residual<M>>),
    MagsacPlusPlus(SigmaConsensusScoring<M, Residual<M>>),
}

impl<M> Scoring<M> for ApiScoring<M> {
    type Score = Score;

    fn threshold(&self) -> f64 {
        match self {
            Self::Ransac(scoring) => scoring.threshold(),
            Self::Msac(scoring) => scoring.threshold(),
            Self::Magsac(scoring) => scoring.threshold(),
            Self::MagsacPlusPlus(scoring) => scoring.threshold(),
        }
    }

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Score {
        match self {
            Self::Ransac(scoring) => scoring.score(data, model, inliers_out),
            Self::Msac(scoring) => scoring.score(data, model, inliers_out),
            Self::Magsac(scoring) => scoring.score(data, model, inliers_out),
            Self::MagsacPlusPlus(scoring) => scoring.score(data, model, inliers_out),
        }
    }
}

fn api_scoring<M>(
    scoring_type: ScoringType,
    threshold: f64,
    degrees_of_freedom: usize,
    priors: Option<&[f64]>,
    residual: Residual<M>,
) -> Result<ApiScoring<M>, String> {
    let scoring = match scoring_type {
        ScoringType::Ransac => {
            let scoring = RansacInlierCountScoring::new(threshold, residual);
            ApiScoring::Ransac(match priors {
                Some(priors) => scoring.with_priors(priors),
                None => scoring,
            })
        }
        ScoringType::Msac => {
            let scoring = MsacScoring::new(threshold, residual);
            ApiScoring::Msac(match priors {
                Some(priors) => scoring.with_priors(priors),
                None => scoring,
            })
        }
        ScoringType::Magsac => {
            let scoring = MagsacScoring::new(threshold, residual)
                .with_sigma_max(threshold)
                .with_degrees_of_freedom(degrees_of_freedom);
            ApiScoring::Magsac(match priors {
                Some(priors) => scoring.with_priors(priors),
                None => scoring,
            })
        }
        ScoringType::MagsacPlusPlus => {
            // SigmaConsensusScoring uses k * sigma_max as its inlier cutoff.
            let scoring =
                SigmaConsensusScoring::new(threshold / 3.64, degrees_of_freedom, residual);
            ApiScoring::MagsacPlusPlus(match priors {
                Some(priors) => scoring.with_priors(priors),
                None => scoring,
            })
        }
        unsupported => {
            return Err(format!(
                "scoring mode {unsupported:?} is not implemented by the high-level API"
            ));
        }
    };
    Ok(scoring)
}

fn api_sampler(settings: &MetasacSettings) -> Result<SamplerChoice, String> {
    match settings.sampler {
        SamplerType::Uniform => Ok(SamplerChoice::Uniform(UniformRandomSampler::new(
            settings.rng_seed,
        ))),
        SamplerType::Prosac => Ok(SamplerChoice::Prosac(ProsacSampler::new(
            100_000,
            settings.rng_seed,
        ))),
        unsupported => Err(format!(
            "sampler {unsupported:?} is not implemented by the high-level API; use Uniform or Prosac"
        )),
    }
}

fn api_optimizer<E>(
    optimization: LocalOptimizationType,
    estimator: E,
) -> Result<Option<LocalOptimizerChoice<E::Model, Score>>, String>
where
    E: Estimator + Send + Sync + 'static,
    E::Model: Clone + Send + Sync + 'static,
{
    match optimization {
        LocalOptimizationType::None => Ok(None),
        LocalOptimizationType::Lsq => Ok(Some(LocalOptimizerChoice::Dyn(Box::new(
            LeastSquaresOptimizer::new(estimator),
        )))),
        unsupported => Err(format!(
            "local optimization {unsupported:?} is not implemented by the high-level API; use None or Lsq"
        )),
    }
}

fn homography_residual(data: &DataMatrix, model: &Homography, idx: usize) -> f64 {
    let p1 = Vector2::new(data.get(idx, 0), data.get(idx, 1));
    let p2 = Vector2::new(data.get(idx, 2), data.get(idx, 3));
    let p1_home = Vector3::new(p1.x, p1.y, 1.0);
    let p2_home = Vector3::new(p2.x, p2.y, 1.0);
    let p2_pred = model.h * p1_home;
    let Some(inverse) = model.h.try_inverse() else {
        return f64::MAX;
    };
    let p1_pred = inverse * p2_home;
    if p2_pred.z.abs() < 1e-12 || p1_pred.z.abs() < 1e-12 {
        return f64::MAX;
    }
    let err1 = (p2 - Vector2::new(p2_pred.x / p2_pred.z, p2_pred.y / p2_pred.z)).norm();
    let err2 = (p1 - Vector2::new(p1_pred.x / p1_pred.z, p1_pred.y / p1_pred.z)).norm();
    (err1 + err2) / 2.0
}

fn fundamental_residual(data: &DataMatrix, model: &FundamentalMatrix, idx: usize) -> f64 {
    let p1 = Vector2::new(data.get(idx, 0), data.get(idx, 1));
    let p2 = Vector2::new(data.get(idx, 2), data.get(idx, 3));
    crate::bundle_adjustment::sampson_error(&model.f, &p1, &p2)
}

fn essential_residual(data: &DataMatrix, model: &EssentialMatrix, idx: usize) -> f64 {
    let p1 = Vector2::new(data.get(idx, 0), data.get(idx, 1));
    let p2 = Vector2::new(data.get(idx, 2), data.get(idx, 3));
    crate::bundle_adjustment::sampson_error(&model.e, &p1, &p2)
}

fn absolute_pose_residual(data: &DataMatrix, model: &AbsolutePose, idx: usize) -> f64 {
    let p_2d = Vector2::new(data.get(idx, 0), data.get(idx, 1));
    let p_3d = Vector3::new(data.get(idx, 2), data.get(idx, 3), data.get(idx, 4));
    crate::bundle_adjustment::reprojection_error(
        model.rotation.to_rotation_matrix().matrix(),
        &model.translation.vector,
        &p_2d,
        &p_3d,
    )
}

fn line_residual(data: &DataMatrix, model: &Line, idx: usize) -> f64 {
    model.distance_to_point(data.get(idx, 0), data.get(idx, 1))
}

fn rigid_transform_residual(data: &DataMatrix, model: &RigidTransform, idx: usize) -> f64 {
    let p1 = nalgebra::Point3::new(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2));
    let p2 = Vector3::new(data.get(idx, 3), data.get(idx, 4), data.get(idx, 5));
    let p1_rot = model.rotation.transform_point(&p1);
    (p2 - (p1_rot.coords + model.translation.vector)).norm()
}

fn plane_residual(data: &DataMatrix, model: &Plane3, idx: usize) -> f64 {
    model.distance(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2))
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
    points1: &DataMatrix,
    points2: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<Homography>, String> {
    validate_threshold(threshold)?;
    if points1.n_points() != points2.n_points() {
        return Err("points1 and points2 must have the same number of points".to_string());
    }
    if points1.n_dims() != 2 || points2.n_dims() != 2 {
        return Err("points must be Nx2 matrices".to_string());
    }
    validate_finite_matrix(points1, "points1")?;
    validate_finite_matrix(points2, "points2")?;
    if !has_non_collinear_2d_points(points1) || !has_non_collinear_2d_points(points2) {
        return Err(
            "homography correspondences must contain non-collinear points in both images"
                .to_string(),
        );
    }

    // Combine into data matrix: [x1, y1, x2, y2]
    let n = points1.n_points();
    let mut data = DataMatrix::zeros(n, 4);
    for i in 0..n {
        data.set(i, 0, points1.get(i, 0));
        data.set(i, 1, points1.get(i, 1));
        data.set(i, 2, points2.get(i, 0));
        data.set(i, 3, points2.get(i, 1));
    }

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = HomographyEstimator::new();
    let sampler = api_sampler(&settings)?;
    let scoring = api_scoring(
        settings.scoring,
        threshold,
        2,
        settings.point_priors.as_deref(),
        homography_residual,
    )?;
    let local_optimizer = api_optimizer(settings.local_optimization, HomographyEstimator::new())?;
    let final_optimizer = api_optimizer(settings.final_optimization, HomographyEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = MetaSAC::new(
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
    points1: &DataMatrix,
    points2: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<FundamentalMatrix>, String> {
    validate_threshold(threshold)?;
    if points1.n_points() != points2.n_points() {
        return Err("points1 and points2 must have the same number of points".to_string());
    }
    if points1.n_dims() != 2 || points2.n_dims() != 2 {
        return Err("points must be Nx2 matrices".to_string());
    }
    validate_finite_matrix(points1, "points1")?;
    validate_finite_matrix(points2, "points2")?;

    // Combine into data matrix: [x1, y1, x2, y2]
    let n = points1.n_points();
    let mut data = DataMatrix::zeros(n, 4);
    for i in 0..n {
        data.set(i, 0, points1.get(i, 0));
        data.set(i, 1, points1.get(i, 1));
        data.set(i, 2, points2.get(i, 0));
        data.set(i, 3, points2.get(i, 1));
    }

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = FundamentalEstimator::new();
    let sampler = api_sampler(&settings)?;
    let scoring = api_scoring(
        settings.scoring,
        threshold,
        1,
        settings.point_priors.as_deref(),
        fundamental_residual,
    )?;
    let local_optimizer = api_optimizer(settings.local_optimization, FundamentalEstimator::new())?;
    let final_optimizer = api_optimizer(settings.final_optimization, FundamentalEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = MetaSAC::new(
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
    points1: &DataMatrix,
    points2: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<EssentialMatrix>, String> {
    validate_threshold(threshold)?;
    if points1.n_points() != points2.n_points() {
        return Err("points1 and points2 must have the same number of points".to_string());
    }
    if points1.n_dims() != 2 || points2.n_dims() != 2 {
        return Err("points must be Nx2 matrices".to_string());
    }
    validate_finite_matrix(points1, "points1")?;
    validate_finite_matrix(points2, "points2")?;

    // Combine into data matrix: [x1, y1, x2, y2]
    let n = points1.n_points();
    let mut data = DataMatrix::zeros(n, 4);
    for i in 0..n {
        data.set(i, 0, points1.get(i, 0));
        data.set(i, 1, points1.get(i, 1));
        data.set(i, 2, points2.get(i, 0));
        data.set(i, 3, points2.get(i, 1));
    }

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = EssentialEstimator::new();
    let sampler = api_sampler(&settings)?;
    let scoring = api_scoring(
        settings.scoring,
        threshold,
        1,
        settings.point_priors.as_deref(),
        essential_residual,
    )?;
    let local_optimizer = api_optimizer(settings.local_optimization, EssentialEstimator::new())?;
    let final_optimizer = api_optimizer(settings.final_optimization, EssentialEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = MetaSAC::new(
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
    points_3d: &DataMatrix,
    points_2d: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<AbsolutePose>, String> {
    validate_threshold(threshold)?;
    if points_3d.n_points() != points_2d.n_points() {
        return Err("points_3d and points_2d must have the same number of points".to_string());
    }
    if points_3d.n_dims() != 3 || points_2d.n_dims() != 2 {
        return Err("points_3d must be Nx3 and points_2d must be Nx2".to_string());
    }
    validate_finite_matrix(points_3d, "points_3d")?;
    validate_finite_matrix(points_2d, "points_2d")?;

    // Combine into data matrix: [x2d, y2d, x3d, y3d, z3d]
    let n = points_3d.n_points();
    let mut data = DataMatrix::zeros(n, 5);
    for i in 0..n {
        data.set(i, 0, points_2d.get(i, 0));
        data.set(i, 1, points_2d.get(i, 1));
        data.set(i, 2, points_3d.get(i, 0));
        data.set(i, 3, points_3d.get(i, 1));
        data.set(i, 4, points_3d.get(i, 2));
    }

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = AbsolutePoseEstimator::new();
    let sampler = api_sampler(&settings)?;
    let scoring = api_scoring(
        settings.scoring,
        threshold,
        2,
        settings.point_priors.as_deref(),
        absolute_pose_residual,
    )?;
    let local_optimizer = api_optimizer(settings.local_optimization, AbsolutePoseEstimator::new())?;
    let final_optimizer = api_optimizer(settings.final_optimization, AbsolutePoseEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = MetaSAC::new(
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
/// use inlier::settings::MetasacSettings;
/// use inlier::types::DataMatrix;
///
/// // Generate some 2D points
/// let mut points = DataMatrix::zeros(10, 2);
/// points.set(0, 0, 0.0); points.set(0, 1, 0.0);
/// points.set(1, 0, 1.0); points.set(1, 1, 1.0);
/// // ... add more points
///
/// let threshold = 0.5; // Distance threshold
/// let settings = MetasacSettings::default();
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
    points: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<Line>, String> {
    validate_threshold(threshold)?;
    if points.n_dims() != 2 {
        return Err("points must be Nx2 matrix (each row is [x, y])".to_string());
    }

    // Use points directly as data matrix
    let n = points.n_points();
    if n < 2 {
        return Err("need at least 2 points to fit a line".to_string());
    }
    validate_finite_matrix(points, "points")?;
    let mut data = DataMatrix::zeros(n, 2);
    for i in 0..n {
        data.set(i, 0, points.get(i, 0));
        data.set(i, 1, points.get(i, 1));
    }

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = LineEstimator::new();
    let sampler = api_sampler(&settings)?;
    let scoring = api_scoring(
        settings.scoring,
        threshold,
        1,
        settings.point_priors.as_deref(),
        line_residual,
    )?;
    let local_optimizer = api_optimizer(settings.local_optimization, LineEstimator::new())?;
    let final_optimizer = api_optimizer(settings.final_optimization, LineEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };
    let inlier_selector = NoopInlierSelector;

    let mut ransac = MetaSAC::new(
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

/// Estimate a rigid transform (rotation + translation) from 3D-3D correspondences.
///
/// Expects an Nx3 source matrix and an Nx3 target matrix.
pub fn estimate_rigid_transform(
    points_src: &DataMatrix,
    points_tgt: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<RigidTransform>, String> {
    validate_threshold(threshold)?;
    if points_src.n_points() != points_tgt.n_points() {
        return Err("points_src and points_tgt must have the same number of points".to_string());
    }
    if points_src.n_dims() != 3 || points_tgt.n_dims() != 3 {
        return Err("points must be Nx3 matrices".to_string());
    }

    let n = points_src.n_points();
    if n < 3 {
        return Err("need at least 3 correspondences to fit a rigid transform".to_string());
    }
    validate_finite_matrix(points_src, "points_src")?;
    validate_finite_matrix(points_tgt, "points_tgt")?;
    let mut data = DataMatrix::zeros(n, 6);
    for i in 0..n {
        data.set(i, 0, points_src.get(i, 0));
        data.set(i, 1, points_src.get(i, 1));
        data.set(i, 2, points_src.get(i, 2));
        data.set(i, 3, points_tgt.get(i, 0));
        data.set(i, 4, points_tgt.get(i, 1));
        data.set(i, 5, points_tgt.get(i, 2));
    }

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = RigidTransformEstimator::new();
    let sampler = api_sampler(&settings)?;

    let scoring = api_scoring(
        settings.scoring,
        threshold,
        3,
        settings.point_priors.as_deref(),
        rigid_transform_residual,
    )?;

    let local_optimizer =
        api_optimizer(settings.local_optimization, RigidTransformEstimator::new())?;
    let final_optimizer =
        api_optimizer(settings.final_optimization, RigidTransformEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };

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

/// Fit a 3-D plane to a point cloud using RANSAC.
///
/// # Arguments
/// * `points` — Nx3 matrix (each row is `[x, y, z]`)
/// * `threshold` — inlier distance threshold in the same units as the point cloud
/// * `settings` — optional RANSAC settings; `None` uses defaults (MSAC scoring)
///
/// # Returns
/// `EstimationResult` with the best [`Plane3`], inlier indices, score, and iteration count.
pub fn estimate_plane(
    points: &DataMatrix,
    threshold: f64,
    settings_opt: Option<MetasacSettings>,
) -> Result<EstimationResult<Plane3>, String> {
    validate_threshold(threshold)?;
    if points.n_dims() != 3 {
        return Err("points must be an Nx3 matrix (each row is [x, y, z])".to_string());
    }
    let n = points.n_points();
    if n < 3 {
        return Err("need at least 3 points to fit a plane".to_string());
    }
    validate_finite_matrix(points, "points")?;

    let settings = settings_opt.unwrap_or_default();
    validate_settings(&settings, n)?;
    let estimator = PlaneEstimator::new();

    let sampler = api_sampler(&settings)?;

    let scoring = api_scoring(
        settings.scoring,
        threshold,
        1,
        settings.point_priors.as_deref(),
        plane_residual,
    )?;

    let local_optimizer = api_optimizer(settings.local_optimization, PlaneEstimator::new())?;
    let final_optimizer = api_optimizer(settings.final_optimization, PlaneEstimator::new())?;
    let termination = crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    };

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

    ransac.run(points);

    match (&ransac.best_model, &ransac.best_score) {
        (Some(model), Some(score)) => Ok(EstimationResult {
            model: model.clone(),
            inliers: ransac.best_inliers.clone(),
            score: *score,
            iterations: ransac.iteration,
        }),
        _ => Err("Failed to estimate plane".to_string()),
    }
}
