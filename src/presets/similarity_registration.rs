//! Similarity (scale + rotation + translation) registration preset inspired by TEASER.

use crate::choices::{LocalOptimizerChoice, SamplerChoice, default_termination};
use crate::core::NoopInlierSelector;
use crate::estimators::SimilarityTransformEstimator;
use crate::models::SimilarityTransform;
use crate::optimisers::LeastSquaresOptimizer;
use crate::pipeline::{CorePipeline, Pipeline};
use crate::preconditioner::IdentityPreconditioner;
use crate::samplers::{ProsacSampler, UniformRandomSampler};
use crate::scoring::{MsacScoring, Score};
use crate::settings::MetasacSettings;
use nalgebra::Vector3;

/// Build a preset pipeline for similarity registration (scale + rotation + translation).
pub fn similarity_registration_pipeline(
    threshold: f64,
    has_priors: bool,
    rng_seed: Option<u64>,
    settings: MetasacSettings,
) -> impl Pipeline<Model = SimilarityTransform, Score = Score> {
    let estimator = SimilarityTransformEstimator::new();

    let sampler = if has_priors {
        SamplerChoice::Prosac(ProsacSampler::new(100_000, rng_seed))
    } else {
        SamplerChoice::Uniform(UniformRandomSampler::new(rng_seed))
    };

    let scoring = MsacScoring::new(threshold, |data, model: &SimilarityTransform, idx| {
        let p1 = nalgebra::Point3::new(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2));
        let p2 = Vector3::new(data.get(idx, 3), data.get(idx, 4), data.get(idx, 5));
        let p1_rotated = model.rotation.transform_point(&p1);
        let p1_final_vec = model.scale * p1_rotated.coords + model.translation.vector;
        (p2 - p1_final_vec).norm()
    });

    // Least squares optimizer can refine similarity as well.
    let local_optimizer = LocalOptimizerChoice::Dyn(Box::new(LeastSquaresOptimizer::new(
        SimilarityTransformEstimator::new(),
    )));
    let final_optimizer = LocalOptimizerChoice::Dyn(Box::new(LeastSquaresOptimizer::new(
        SimilarityTransformEstimator::new(),
    )));

    let termination = default_termination(&settings);

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
