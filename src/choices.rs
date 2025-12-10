//! Lightweight runtime wrappers exposing built-in components via enums while the
//! core `SuperRansac` stays fully generic. Each enum holds concrete variants
//! plus a `Dyn` escape hatch for custom implementations.

use crate::core::{InlierSelector, LocalOptimizer, Sampler, TerminationCriterion};
use crate::samplers::{ProsacSampler, UniformRandomSampler};
use crate::scoring::Score;
use crate::settings::RansacSettings;
use crate::types::DataMatrix;

/// Runtime sampler selection.
pub enum SamplerChoice {
    Uniform(UniformRandomSampler),
    Prosac(ProsacSampler),
    Dyn(Box<dyn Sampler + Send + Sync>),
}

impl Default for SamplerChoice {
    fn default() -> Self {
        SamplerChoice::Prosac(ProsacSampler::new())
    }
}

impl Sampler for SamplerChoice {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
        match self {
            SamplerChoice::Uniform(s) => s.sample(data, sample_size, out_indices),
            SamplerChoice::Prosac(s) => s.sample(data, sample_size, out_indices),
            SamplerChoice::Dyn(s) => s.sample(data, sample_size, out_indices),
        }
    }

    fn update(&mut self, sample: &[usize], sample_size: usize, iteration: usize, score_hint: f64) {
        match self {
            SamplerChoice::Uniform(s) => s.update(sample, sample_size, iteration, score_hint),
            SamplerChoice::Prosac(s) => s.update(sample, sample_size, iteration, score_hint),
            SamplerChoice::Dyn(s) => s.update(sample, sample_size, iteration, score_hint),
        }
    }
}

/// Runtime termination selection (Score-specific).
pub enum TerminationChoice {
    Ransac(crate::core::RansacTerminationCriterion),
    Dyn(Box<dyn TerminationCriterion<Score> + Send + Sync>),
}

impl TerminationCriterion<Score> for TerminationChoice {
    fn check(
        &mut self,
        data: &DataMatrix,
        best_score: &Score,
        sample_size: usize,
        max_iterations: &mut usize,
    ) -> bool {
        match self {
            TerminationChoice::Ransac(term) => {
                term.check(data, best_score, sample_size, max_iterations)
            }
            TerminationChoice::Dyn(term) => {
                term.check(data, best_score, sample_size, max_iterations)
            }
        }
    }
}

/// Runtime local optimizer selection.
pub enum LocalOptimizerChoice<M, S> {
    None,
    Dyn(Box<dyn LocalOptimizer<M, S> + Send + Sync>),
}

impl<M, S> LocalOptimizer<M, S> for LocalOptimizerChoice<M, S>
where
    M: Clone,
    S: Clone,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &M,
        best_score: &S,
    ) -> (M, S, Vec<usize>) {
        match self {
            LocalOptimizerChoice::None => (model.clone(), best_score.clone(), inliers.to_vec()),
            LocalOptimizerChoice::Dyn(opt) => opt.run(data, inliers, model, best_score),
        }
    }
}

/// Runtime inlier selector selection.
pub enum InlierSelectorChoice<M> {
    None,
    Dyn(Box<dyn InlierSelector<M> + Send + Sync>),
}

impl<M> InlierSelector<M> for InlierSelectorChoice<M> {
    fn select(&mut self, data: &DataMatrix, model: &M) -> Vec<usize> {
        match self {
            InlierSelectorChoice::None => (0..data.nrows()).collect(),
            InlierSelectorChoice::Dyn(sel) => sel.select(data, model),
        }
    }
}

/// Helper to build a `TerminationChoice` from settings.
pub fn default_termination(settings: &RansacSettings) -> TerminationChoice {
    TerminationChoice::Ransac(crate::core::RansacTerminationCriterion {
        confidence: settings.confidence,
    })
}
