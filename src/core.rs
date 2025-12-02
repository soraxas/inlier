//! Core SupeRANSAC traits and pipeline skeleton.
//!
//! The goal of this module is to express the architecture of the original
//! C++ SupeRANSAC (`superansac::SupeRansac`) in idiomatic Rust. The actual
//! numerical routines are implemented in later phases; here we focus on:
//! - Traits for estimators, samplers, scoring, local optimization, termination,
//!   and inlier selection.
//! - A generic `SuperRansac` struct that orchestrates these components.

use crate::{settings::RansacSettings, types::DataMatrix};

/// Estimator responsible for generating model hypotheses from minimal samples.
pub trait Estimator {
    /// Model type produced by this estimator.
    type Model: Clone;

    /// Size of a minimal sample for this estimator.
    fn sample_size(&self) -> usize;

    /// Check whether a given sample is geometrically valid.
    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool;

    /// Estimate candidate models from a minimal sample.
    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model>;

    /// Validate a candidate model before scoring.
    fn is_valid_model(
        &self,
        model: &Self::Model,
        data: &DataMatrix,
        sample: &[usize],
        threshold: f64,
    ) -> bool;
}

/// Sampler responsible for drawing minimal samples from the data.
pub trait Sampler {
    /// Draw a sample of `sample_size` elements into `out_indices`.
    ///
    /// Returns `false` if a valid sample could not be drawn (caller may retry).
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool;

    /// Update the sampler state given the last sample and iteration.
    fn update(&mut self, sample: &[usize], sample_size: usize, iteration: usize, score_hint: f64);
}

/// Scoring strategy used to evaluate model quality and determine inliers.
pub trait Scoring<M> {
    /// Score type – must support ordering for "better than" comparisons.
    type Score: Clone + PartialOrd;

    /// Inlier/outlier threshold for residuals in the chosen domain.
    fn threshold(&self) -> f64;

    /// Score a model and optionally return the inlier set.
    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score;
}

/// Local optimization strategy, potentially refining a model using its inliers.
pub trait LocalOptimizer<M, S: Clone> {
    /// Run local optimization on the given model and inliers.
    ///
    /// Returns `(refined_model, refined_score, refined_inliers)`.
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &M,
        best_score: &S,
    ) -> (M, S, Vec<usize>);
}

/// Termination criterion deciding when the RANSAC loop can stop.
pub trait TerminationCriterion<S> {
    /// Update the termination state.
    ///
    /// Returns `true` if the algorithm should terminate immediately.
    fn check(
        &mut self,
        data: &DataMatrix,
        best_score: &S,
        sample_size: usize,
        max_iterations: &mut usize,
    ) -> bool;
}

/// Optional inlier selector (e.g. space-partitioning RANSAC).
pub trait InlierSelector<M> {
    /// Select a (possibly reduced) set of inliers to consider during scoring.
    fn select(&mut self, data: &DataMatrix, model: &M) -> Vec<usize>;
}

/// Generic SupeRANSAC pipeline orchestrating the above components.
///
/// This is a structural analogue of the C++ `superansac::SupeRansac` class.
#[derive(Debug)]
pub struct SuperRansac<E, Sa, Sc, LO, T, IS>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    Sc::Score: Clone + PartialOrd,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion<Sc::Score>,
    IS: InlierSelector<E::Model>,
{
    pub settings: RansacSettings,
    pub estimator: E,
    pub sampler: Sa,
    pub scoring: Sc,
    pub local_optimizer: Option<LO>,
    pub final_optimizer: Option<LO>,
    pub termination: T,
    pub inlier_selector: Option<IS>,

    // Outputs / diagnostics
    pub best_model: Option<E::Model>,
    pub best_inliers: Vec<usize>,
    pub best_score: Option<Sc::Score>,
    pub iteration: usize,
}

impl<E, Sa, Sc, LO, T, IS> SuperRansac<E, Sa, Sc, LO, T, IS>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    Sc::Score: Clone + PartialOrd,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion<Sc::Score>,
    IS: InlierSelector<E::Model>,
{
    /// Create a new pipeline from its components.
    pub fn new(
        settings: RansacSettings,
        estimator: E,
        sampler: Sa,
        scoring: Sc,
        local_optimizer: Option<LO>,
        final_optimizer: Option<LO>,
        termination: T,
        inlier_selector: Option<IS>,
    ) -> Self {
        Self {
            settings,
            estimator,
            sampler,
            scoring,
            local_optimizer,
            final_optimizer,
            termination,
            inlier_selector,
            best_model: None,
            best_inliers: Vec::new(),
            best_score: None,
            iteration: 0,
        }
    }

    /// Run the RANSAC loop on the given data matrix.
    ///
    /// This is a simplified but structurally similar version of the C++
    /// `SupeRansac::run` implementation: it manages sampling, model
    /// generation, scoring, (optional) local optimization, and termination.
    pub fn run(&mut self, data: &DataMatrix) {
        let sample_size = self.estimator.sample_size();
        let mut sample = vec![0usize; sample_size];
        let mut tmp_inliers = Vec::new();

        let mut max_iterations = self.settings.max_iterations;
        let min_iterations = self.settings.min_iterations;

        self.best_inliers.clear();
        self.best_model = None;
        self.best_score = None;
        self.iteration = 0;

        let threshold = self.scoring.threshold();

        while self.iteration < max_iterations || self.iteration < min_iterations {
            // Try to obtain a valid sample and a valid model.
            let mut have_model = false;
            let mut models: Vec<E::Model> = Vec::new();

            for _ in 0..100 {
                if !self.sampler.sample(data, sample_size, &mut sample[..]) {
                    self.sampler
                        .update(&sample, sample_size, self.iteration, 0.0);
                    continue;
                }

                if !self.estimator.is_valid_sample(data, &sample) {
                    self.sampler
                        .update(&sample, sample_size, self.iteration, 0.0);
                    continue;
                }

                models = self.estimator.estimate_model(data, &sample);
                if models.is_empty() {
                    self.sampler
                        .update(&sample, sample_size, self.iteration, 0.0);
                    continue;
                }

                have_model = true;
                break;
            }

            if !have_model {
                // No valid model could be generated for this iteration.
                self.iteration += 1;
                continue;
            }

            // Evaluate each model.
            let mut iteration_improved_best = false;

            for model in models.iter() {
                if !self
                    .estimator
                    .is_valid_model(model, data, &sample, threshold)
                {
                    continue;
                }

                tmp_inliers.clear();

                // Optionally let an inlier selector narrow down the points.
                let _selected_inliers = if let Some(selector) = &mut self.inlier_selector {
                    selector.select(data, model)
                } else {
                    Vec::new()
                };

                let score = self.scoring.score(data, model, &mut tmp_inliers);

                let better = match &self.best_score {
                    None => true,
                    Some(best) => score > *best,
                };

                if better {
                    self.best_score = Some(score.clone());
                    self.best_model = Some(model.clone());
                    self.best_inliers.clear();
                    self.best_inliers.extend_from_slice(&tmp_inliers);
                    iteration_improved_best = true;
                }
            }

            // Local optimization if we improved this iteration.
            if iteration_improved_best {
                if let (Some(lo), Some(best_model), Some(best_score)) = (
                    &mut self.local_optimizer,
                    &self.best_model,
                    &self.best_score,
                ) {
                    let lo = lo;
                    let best_model = best_model;
                    let best_score = best_score;
                    let (refined_model, refined_score, refined_inliers) =
                        lo.run(data, &self.best_inliers, best_model, best_score);

                    if refined_score > *best_score {
                        self.best_model = Some(refined_model);
                        self.best_score = Some(refined_score);
                        self.best_inliers = refined_inliers;
                    }
                }

                // Update termination criterion using the current best score.
                if let (Some(best_score), Some(_best_model)) = (&self.best_score, &self.best_model)
                {
                    let best_score = best_score;
                    let should_terminate =
                        self.termination
                            .check(data, best_score, sample_size, &mut max_iterations);
                    if should_terminate {
                        break;
                    }
                }
            }

            // Sampler update at the end of the iteration.
            self.sampler
                .update(&sample, sample_size, self.iteration, 0.0);

            self.iteration += 1;
        }

        // Optional final optimization step once iterations are done.
        if let (Some(final_opt), Some(best_model), Some(best_score)) = (
            &mut self.final_optimizer,
            &self.best_model,
            &self.best_score,
        ) {
            let final_opt = final_opt;
            let best_model = best_model;
            let best_score = best_score;
            if self.best_inliers.len() > sample_size {
                let (refined_model, refined_score, refined_inliers) =
                    final_opt.run(data, &self.best_inliers, best_model, best_score);

                if refined_score > *best_score {
                    self.best_model = Some(refined_model);
                    self.best_score = Some(refined_score);
                    self.best_inliers = refined_inliers;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::settings::RansacSettings;
    use crate::types::DataMatrix;

    #[derive(Clone, Debug)]
    struct MockModel;

    #[derive(Clone, Debug, PartialEq, PartialOrd)]
    struct MockScore(f64);

    struct MockEstimator;

    impl Estimator for MockEstimator {
        type Model = MockModel;

        fn sample_size(&self) -> usize {
            2
        }

        fn is_valid_sample(&self, _data: &DataMatrix, _sample: &[usize]) -> bool {
            true
        }

        fn estimate_model(&self, _data: &DataMatrix, _sample: &[usize]) -> Vec<Self::Model> {
            vec![MockModel]
        }

        fn is_valid_model(
            &self,
            _model: &Self::Model,
            _data: &DataMatrix,
            _sample: &[usize],
            _threshold: f64,
        ) -> bool {
            true
        }
    }

    struct MockSampler {
        pub sample_calls: usize,
        pub update_calls: usize,
    }

    impl Sampler for MockSampler {
        fn sample(
            &mut self,
            data: &DataMatrix,
            sample_size: usize,
            out_indices: &mut [usize],
        ) -> bool {
            self.sample_calls += 1;
            // Always return the first `sample_size` indices.
            for (i, v) in out_indices.iter_mut().enumerate().take(sample_size) {
                *v = i.min(data.nrows().saturating_sub(1));
            }
            true
        }

        fn update(
            &mut self,
            _sample: &[usize],
            _sample_size: usize,
            _iteration: usize,
            _score_hint: f64,
        ) {
            self.update_calls += 1;
        }
    }

    struct MockScoring;

    impl Scoring<MockModel> for MockScoring {
        type Score = MockScore;

        fn threshold(&self) -> f64 {
            1.0
        }

        fn score(
            &self,
            _data: &DataMatrix,
            _model: &MockModel,
            inliers_out: &mut Vec<usize>,
        ) -> Self::Score {
            // Pretend all points are inliers with a fixed score.
            inliers_out.clear();
            inliers_out.push(0);
            MockScore(1.0)
        }
    }

    struct MockLocalOptimizer;

    impl LocalOptimizer<MockModel, MockScore> for MockLocalOptimizer {
        fn run(
            &mut self,
            _data: &DataMatrix,
            inliers: &[usize],
            model: &MockModel,
            best_score: &MockScore,
        ) -> (MockModel, MockScore, Vec<usize>) {
            // Return the same model and score, with the same inliers.
            (model.clone(), best_score.clone(), inliers.to_vec())
        }
    }

    struct MockTerminationCriterion {
        pub called: usize,
    }

    impl TerminationCriterion<MockScore> for MockTerminationCriterion {
        fn check(
            &mut self,
            _data: &DataMatrix,
            _best_score: &MockScore,
            _sample_size: usize,
            max_iterations: &mut usize,
        ) -> bool {
            self.called += 1;
            // After the first call, cap the maximum iterations to the current iteration + 1.
            *max_iterations = (*max_iterations).min(1);
            true
        }
    }

    struct MockInlierSelector;

    impl InlierSelector<MockModel> for MockInlierSelector {
        fn select(&mut self, _data: &DataMatrix, _model: &MockModel) -> Vec<usize> {
            vec![0]
        }
    }

    #[test]
    fn superransac_runs_and_updates_best_model() {
        let data = DataMatrix::zeros(4, 2);
        let settings = RansacSettings::default();

        let estimator = MockEstimator;
        let sampler = MockSampler {
            sample_calls: 0,
            update_calls: 0,
        };
        let scoring = MockScoring;
        let local_optimizer = Some(MockLocalOptimizer);
        let final_optimizer = None;
        let termination = MockTerminationCriterion { called: 0 };
        let inlier_selector = Some(MockInlierSelector);

        let mut pipeline = SuperRansac::new(
            settings,
            estimator,
            sampler,
            scoring,
            local_optimizer,
            final_optimizer,
            termination,
            inlier_selector,
        );

        pipeline.run(&data);

        assert!(
            pipeline.best_model.is_some(),
            "Best model should be set after run"
        );
        assert!(
            pipeline.best_score.is_some(),
            "Best score should be set after run"
        );
        assert!(
            !pipeline.best_inliers.is_empty(),
            "Best inliers should not be empty"
        );
        // We don't assert an exact iteration count here, only that `run`
        // completed without panicking and produced a consistent result.
    }
}
