//! TEASER-inspired rigid 3D registration preset.
//!
//! This mirrors the rigid transform preset but uses a dedicated estimator handle and
//! a truncated-loss style scoring (MSAC) to better mirror TEASER’s maximum-consensus view.
//! It keeps the code simple while making the sampling/optimization choices explicit.

use crate::choices::{default_termination, InlierSelectorChoice, LocalOptimizerChoice, SamplerChoice};
use crate::core::{Estimator, InlierSelector, LocalOptimizer, NoopInlierSelector, Sampler, Scoring};
use crate::estimators::RigidTransformEstimator;
use crate::models::RigidTransform;
use crate::optimisers::LeastSquaresOptimizer;
use crate::pipeline::{Pipeline, PipelineBuilder, Preconditioner};
use crate::samplers::{ProsacSampler, UniformRandomSampler};
use crate::scoring::{MsacScoring, Score};
use crate::settings::MetasacSettings;
use crate::types::DataMatrix;
use nalgebra::Vector3;

/// Identity preconditioner; leaves data and model unchanged.
#[derive(Clone)]
struct IdentityRigidPreconditioner;

impl Preconditioner<RigidTransform> for IdentityRigidPreconditioner {
    type Normalization = ();

    fn normalize(&self, data: &DataMatrix) -> (DataMatrix, Self::Normalization) {
        (data.clone(), ())
    }

    fn denormalize(&self, model: &RigidTransform, _norm: &Self::Normalization) -> RigidTransform {
        model.clone()
    }
}

/// Thin wrapper around the native estimator so we can hang future graph-pruning/decoupling tweaks here.
#[derive(Clone, Default)]
pub struct TeaserLikeRigidEstimator {
    inner: RigidTransformEstimator,
}

impl TeaserLikeRigidEstimator {
    pub fn new() -> Self {
        Self {
            inner: RigidTransformEstimator::new(),
        }
    }
}

impl Estimator for TeaserLikeRigidEstimator {
    type Model = RigidTransform;

    fn sample_size(&self) -> usize {
        self.inner.sample_size()
    }

    fn non_minimal_sample_size(&self) -> usize {
        self.inner.sample_size()
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        self.inner.is_valid_sample(data, sample)
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        self.inner.estimate_model(data, sample)
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        self.inner
            .estimate_model_nonminimal(data, sample, weights)
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        data: &DataMatrix,
        sample: &[usize],
        threshold: f64,
    ) -> bool {
        self.inner
            .is_valid_model(model, data, sample, threshold)
    }
}

/// Build a preset pipeline for rigid 3D registration using truncated-loss scoring.
pub fn rigid_registration_pipeline(
    threshold: f64,
    has_priors: bool,
    rng_seed: Option<u64>,
    settings: MetasacSettings,
) -> impl Pipeline<Model = RigidTransform, Score = Score> {
    let estimator = TeaserLikeRigidEstimator::new();

    let sampler = if has_priors {
        SamplerChoice::Prosac(ProsacSampler::new(100_000, rng_seed))
    } else {
        SamplerChoice::Uniform(UniformRandomSampler::new(rng_seed))
    };

    let scoring = MsacScoring::new(threshold, |data, model: &RigidTransform, idx| {
        // Point-to-point residual
        let p1 = nalgebra::Point3::new(data[(idx, 0)], data[(idx, 1)], data[(idx, 2)]);
        let p2 = Vector3::new(data[(idx, 3)], data[(idx, 4)], data[(idx, 5)]);
        let p1_rotated = model.rotation.transform_point(&p1);
        let p1_final_vec = p1_rotated.coords + model.translation.vector;
        (p2 - p1_final_vec).norm()
    });

    let local_optimizer = LocalOptimizerChoice::Dyn(Box::new(LeastSquaresOptimizer::new(
        RigidTransformEstimator::new(),
    )));
    let final_optimizer = LocalOptimizerChoice::Dyn(Box::new(LeastSquaresOptimizer::new(
        RigidTransformEstimator::new(),
    )));

    let termination = default_termination(&settings);

    PipelineBuilder::new(settings, estimator, sampler, scoring, termination)
        .with_local_optimizer(local_optimizer)
        .with_final_optimizer(final_optimizer)
        .with_inlier_selector(InlierSelectorChoice::Dyn(Box::new(NoopInlierSelector)))
        .with_preconditioner(IdentityRigidPreconditioner)
}
