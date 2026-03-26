//! Rigid transform preset for SupeRANSAC.
//!
//! This module provides user-friendly functions for estimating geometric models
//! similar to the Python API.

use crate::choices::SamplerChoice;
use crate::core::NoopInlierSelector;
use crate::core::*;
use crate::estimators::RigidTransformEstimator;
use crate::models::RigidTransform;
use crate::optimisers::LeastSquaresOptimizer;
use crate::pipeline::CorePipeline;
use crate::pipeline::Pipeline;
use crate::preconditioner::IdentityPreconditioner;
use crate::samplers::{ProsacSampler, UniformRandomSampler};
use crate::scoring::{RansacInlierCountScoring, Score};
use crate::settings::MetasacSettings;
use nalgebra::Vector3;

/// Build a preset RANSAC pipeline for estimating a rigid 3D-3D transform.
///
/// This constructs a `CorePipeline` with:
/// - `RigidTransformEstimator`
/// - `SamplerChoice` (Prosac or Uniform from `settings`)
/// - `RansacInlierCountScoring` with point-to-point distance
/// - `LeastSquaresOptimizer` as both local and final optimizers
/// - standard RANSAC termination from `settings.confidence`
/// - `NoopInlierSelector`
/// - identity preconditioner
pub fn rigid_transform_pipeline(
    threshold: f64,
    has_priors: bool,
    rng_seed: Option<u64>,
    settings: MetasacSettings,
) -> impl Pipeline<Model = RigidTransform, Score = Score> {
    let estimator = RigidTransformEstimator::new();

    let sampler = if has_priors {
        SamplerChoice::Prosac(ProsacSampler::new(100_000, rng_seed))
    } else {
        SamplerChoice::Uniform(UniformRandomSampler::new(rng_seed))
    };
    let scoring = RansacInlierCountScoring::new(threshold, |data, model: &RigidTransform, idx| {
        // Compute point-to-point distance
        let p1 = nalgebra::Point3::new(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2));
        let p2 = Vector3::new(data.get(idx, 3), data.get(idx, 4), data.get(idx, 5));
        let p1_rotated = model.rotation.transform_point(&p1);
        let p1_final_vec = p1_rotated.coords + model.translation.vector;
        (p2 - p1_final_vec).norm()
    });
    let local_optimizer = LeastSquaresOptimizer::new(RigidTransformEstimator::new());
    let final_optimizer = LeastSquaresOptimizer::new(RigidTransformEstimator::new());
    let termination = RansacTerminationCriterion {
        confidence: settings.confidence,
    };

    CorePipeline::<_, _, _, _, _, NoopInlierSelector, IdentityPreconditioner>::new(
        settings,
        estimator,
        sampler,
        scoring,
        termination,
    )
    .with_local_optimizer(local_optimizer)
    .with_final_optimizer(final_optimizer)
}
