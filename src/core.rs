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

    /// Size of a non-minimal sample required for the estimation.
    /// Defaults to `sample_size()` if not overridden.
    fn non_minimal_sample_size(&self) -> usize {
        self.sample_size()
    }

    /// Check whether a given sample is geometrically valid.
    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool;

    /// Estimate candidate models from a minimal sample.
    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model>;

    /// Estimate candidate models from a non-minimal sample (typically all inliers).
    /// This allows for more robust fitting using least-squares or similar methods.
    /// The `weights` parameter is optional and can be used for weighted least-squares.
    /// Default implementation falls back to `estimate_model`.
    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        _weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        // Default: fall back to minimal estimation
        self.estimate_model(data, sample)
    }

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

/// Trivial local optimizer that returns the input model/score/inliers unchanged.
///
/// This is primarily useful as a placeholder during early porting phases or
/// when local optimization is disabled in the settings.
pub struct NoopLocalOptimizer;

impl<M: Clone, S: Clone> LocalOptimizer<M, S> for NoopLocalOptimizer {
    fn run(
        &mut self,
        _data: &DataMatrix,
        inliers: &[usize],
        model: &M,
        best_score: &S,
    ) -> (M, S, Vec<usize>) {
        (model.clone(), best_score.clone(), inliers.to_vec())
    }
}

/// Least Squares local optimizer that refits the model using all inliers.
///
/// This requires the estimator to support non-minimal fitting (i.e., fitting
/// from more than the minimal sample size). For estimators that don't support
/// this, it falls back to returning the original model.
pub struct LeastSquaresOptimizer<E>
where
    E: Estimator,
{
    estimator: E,
    use_inliers: bool,
}

/// Iterated Least Squares optimizer that performs multiple iterations of
/// refitting, potentially improving the model quality.
pub struct IteratedLeastSquaresOptimizer<E>
where
    E: Estimator,
{
    estimator: E,
    max_iterations: usize,
    use_inliers: bool,
}

impl<E> LeastSquaresOptimizer<E>
where
    E: Estimator,
{
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            use_inliers: true,
        }
    }

    pub fn set_use_inliers(&mut self, use_inliers: bool) {
        self.use_inliers = use_inliers;
    }
}

impl<E, S> LocalOptimizer<E::Model, S> for LeastSquaresOptimizer<E>
where
    E: Estimator,
    E::Model: Clone,
    S: Clone,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &E::Model,
        best_score: &S,
    ) -> (E::Model, S, Vec<usize>) {
        if !self.use_inliers || inliers.len() < self.estimator.non_minimal_sample_size() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        // Refit model using all inliers with non-minimal estimation (like C++)
        let refined_models = self.estimator.estimate_model_nonminimal(data, inliers, None);

        if refined_models.is_empty() {
            // Fallback to original model if refitting fails
            (model.clone(), best_score.clone(), inliers.to_vec())
        } else {
            // Use the first refined model
            (refined_models[0].clone(), best_score.clone(), inliers.to_vec())
        }
    }
}

impl<E> IteratedLeastSquaresOptimizer<E>
where
    E: Estimator,
{
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            max_iterations: 5,
            use_inliers: true,
        }
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    pub fn set_use_inliers(&mut self, use_inliers: bool) {
        self.use_inliers = use_inliers;
    }
}

impl<E, S> LocalOptimizer<E::Model, S> for IteratedLeastSquaresOptimizer<E>
where
    E: Estimator,
    E::Model: Clone,
    S: Clone + PartialOrd,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &E::Model,
        best_score: &S,
    ) -> (E::Model, S, Vec<usize>) {
        if !self.use_inliers || inliers.len() < self.estimator.non_minimal_sample_size() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        // Iteratively refit the model
        let mut current_model = model.clone();
        let mut current_inliers = inliers.to_vec();

        for _iteration in 0..self.max_iterations {
            // Refit using current inliers
            let refined_models = self.estimator.estimate_model_nonminimal(data, &current_inliers, None);

            if refined_models.is_empty() {
                break;
            }

            // In a full implementation, we would:
            // 1. Score the refined model
            // 2. Update inliers based on new residuals
            // 3. Check for convergence
            // For now, we just use the refined model
            current_model = refined_models[0].clone();

            // Simple convergence check: if we can't improve, stop
            // (In practice, we'd check if inliers changed or score improved)
        }

        (current_model, best_score.clone(), current_inliers)
    }
}

/// Nested RANSAC optimizer that runs an inner RANSAC loop on the inliers.
///
/// This performs multiple iterations of sampling from inliers and refitting,
/// potentially finding better models.
pub struct NestedRansacOptimizer<E>
where
    E: Estimator,
{
    estimator: E,
    max_iterations: usize,
    sample_size_multiplier: usize,
}

impl<E> NestedRansacOptimizer<E>
where
    E: Estimator,
{
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            max_iterations: 50,
            sample_size_multiplier: 7,
        }
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    pub fn set_sample_size_multiplier(&mut self, multiplier: usize) {
        self.sample_size_multiplier = multiplier;
    }
}

impl<E, S> LocalOptimizer<E::Model, S> for NestedRansacOptimizer<E>
where
    E: Estimator,
    E::Model: Clone,
    S: Clone + PartialOrd,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &E::Model,
        best_score: &S,
    ) -> (E::Model, S, Vec<usize>) {
        use crate::samplers::UniformRandomSampler;

        if inliers.len() < self.estimator.sample_size() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        let mut current_model = model.clone();
        let current_inliers = inliers.to_vec();
        let current_score = best_score.clone();

        let non_minimal_sample_size = self.sample_size_multiplier * self.estimator.sample_size();
        let mut sampler = UniformRandomSampler::new();

        // Inner RANSAC loop
        for _iteration in 0..self.max_iterations {
            // Calculate current sample size (limited by available inliers)
            let mut current_sample_size = current_inliers.len().saturating_sub(1);
            if current_sample_size >= non_minimal_sample_size {
                current_sample_size = non_minimal_sample_size;
            }

            // Break if sample size is too small
            if current_sample_size < self.estimator.sample_size() {
                break;
            }

            // Sample from inliers
            let mut sample = vec![0usize; current_sample_size];
            if current_sample_size == current_inliers.len() {
                // Use all inliers
                sample.copy_from_slice(&current_inliers[..current_sample_size]);
            } else {
                // Create a temporary data matrix view for sampling from inliers
                // We'll sample indices in [0, current_inliers.len()) and map them
                let mut temp_indices = vec![0usize; current_sample_size];
                // Create a dummy data matrix with the right number of rows for sampling
                let dummy_data = DataMatrix::zeros(current_inliers.len(), data.ncols());
                if !sampler.sample(&dummy_data, current_sample_size, &mut temp_indices) {
                    continue;
                }
                // Map sample indices to actual data point indices
                for (i, &temp_idx) in temp_indices.iter().enumerate() {
                    sample[i] = current_inliers[temp_idx];
                }
            }

            // Estimate model from sample
            let refined_models = self.estimator.estimate_model_nonminimal(data, &sample, None);
            if refined_models.is_empty() {
                continue;
            }

            // In a full implementation, we would score each model and keep the best
            // For now, we just use the first refined model
            // The actual scoring would be done by the caller's scoring function
            current_model = refined_models[0].clone();

            // Re-initialize sampler with new inlier count (simplified)
            // In practice, we'd re-score and update inliers here
        }

        (current_model, current_score, current_inliers)
    }
}

/// Iteratively Reweighted Least Squares (IRLS) local optimizer.
///
/// This performs multiple iterations of weighted least squares, where weights
/// are updated based on residuals from the previous iteration.
pub struct IRLSOptimizer<E>
where
    E: Estimator,
{
    estimator: E,
    max_iterations: usize,
    convergence_threshold: f64,
    use_inliers: bool,
}

impl<E> IRLSOptimizer<E>
where
    E: Estimator,
{
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            max_iterations: 10,
            convergence_threshold: 1e-6,
            use_inliers: true,
        }
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    pub fn set_use_inliers(&mut self, use_inliers: bool) {
        self.use_inliers = use_inliers;
    }
}

impl<E, S> LocalOptimizer<E::Model, S> for IRLSOptimizer<E>
where
    E: Estimator,
    E::Model: Clone,
    S: Clone,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &E::Model,
        best_score: &S,
    ) -> (E::Model, S, Vec<usize>) {
        if !self.use_inliers || inliers.len() < self.estimator.sample_size() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        // Simplified IRLS: iterate a few times with refitting
        // A full implementation would compute weights based on residuals
        let mut current_model = model.clone();
        let weights: Vec<f64> = vec![1.0; inliers.len()];

        for _iteration in 0..self.max_iterations {
            // Use weighted non-minimal estimation (like C++)
            let refined_models = self.estimator.estimate_model_nonminimal(data, inliers, Some(&weights));
            if refined_models.is_empty() {
                break;
            }

            // Check for convergence (simplified: just use the refined model)
            current_model = refined_models[0].clone();

            // In a full implementation, we would:
            // 1. Compute residuals for all inliers
            // 2. Compute weights (e.g., using robust loss functions)
            // 3. Refit with weighted least squares
            // 4. Check convergence
            // For now, we just iterate without weight updates
        }

        (current_model, best_score.clone(), inliers.to_vec())
    }
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

/// Simple RANSAC-style termination criterion that updates the maximum number of
/// iterations using the current best inlier ratio and desired confidence.
///
/// The update rule follows the standard formula
/// `N = log(1 - confidence) / log(1 - inlier_ratio^sample_size)`.
pub struct RansacTerminationCriterion {
    /// Desired confidence in \[0, 1\].
    pub confidence: f64,
}

impl crate::core::TerminationCriterion<crate::scoring::Score> for RansacTerminationCriterion {
    fn check(
        &mut self,
        data: &DataMatrix,
        best_score: &crate::scoring::Score,
        sample_size: usize,
        max_iterations: &mut usize,
    ) -> bool {
        let n = data.nrows() as f64;
        if n <= 0.0 {
            return false;
        }

        let inlier_ratio = (best_score.inlier_count as f64 / n).clamp(0.0, 1.0);
        if inlier_ratio <= 0.0 || inlier_ratio >= 1.0 {
            return false;
        }

        let p_good_sample = inlier_ratio.powi(sample_size as i32);
        if p_good_sample <= 0.0 || p_good_sample >= 1.0 {
            return false;
        }

        let log_one_minus_conf = (1.0 - self.confidence).ln();
        let log_one_minus_p = (1.0 - p_good_sample).ln();
        if !log_one_minus_conf.is_finite() || !log_one_minus_p.is_finite() {
            return false;
        }

        let required = (log_one_minus_conf / log_one_minus_p).ceil().max(1.0) as usize;
        if required < *max_iterations {
            *max_iterations = required;
        }

        // Do not force immediate termination; the outer loop will stop once
        // the (possibly updated) iteration budget is exhausted.
        false
    }
}

/// Optional inlier selector (e.g. space-partitioning RANSAC).
pub trait InlierSelector<M> {
    /// Select a (possibly reduced) set of inliers to consider during scoring.
    fn select(&mut self, data: &DataMatrix, model: &M) -> Vec<usize>;
}

/// Trivial inlier selector that defers entirely to the scoring function by
/// returning an empty pre-selection (i.e. "use all points").
pub struct NoopInlierSelector;

impl<M> InlierSelector<M> for NoopInlierSelector {
    fn select(&mut self, _data: &DataMatrix, _model: &M) -> Vec<usize> {
        Vec::new()
    }
}

/// Space-partitioning inlier selector that pre-filters candidates using
/// spatial partitioning (e.g., grid-based or kd-tree based).
///
/// This is a simplified implementation. A full implementation would use
/// proper spatial data structures like grids or kd-trees.
pub struct SpacePartitioningInlierSelector<F>
where
    F: Fn(&DataMatrix, usize) -> f64,
{
    /// Function to extract spatial coordinate from a data point (e.g., x or y)
    coord_fn: F,
    /// Grid cell size for partitioning
    cell_size: f64,
    /// Minimum number of cells to consider
    min_cells: usize,
}

impl<F> SpacePartitioningInlierSelector<F>
where
    F: Fn(&DataMatrix, usize) -> f64,
{
    pub fn new(coord_fn: F, cell_size: f64) -> Self {
        Self {
            coord_fn,
            cell_size,
            min_cells: 1,
        }
    }

    pub fn set_min_cells(&mut self, min_cells: usize) {
        self.min_cells = min_cells;
    }
}

impl<M, F> InlierSelector<M> for SpacePartitioningInlierSelector<F>
where
    F: Fn(&DataMatrix, usize) -> f64,
{
    fn select(&mut self, data: &DataMatrix, _model: &M) -> Vec<usize> {
        let n = data.nrows();
        if n == 0 {
            return Vec::new();
        }

        // Simplified space partitioning: partition points into grid cells
        // based on their spatial coordinates
        let mut cells: std::collections::HashMap<i32, Vec<usize>> =
            std::collections::HashMap::new();

        for i in 0..n {
            let coord = (self.coord_fn)(data, i);
            let cell_idx = (coord / self.cell_size).floor() as i32;
            cells.entry(cell_idx).or_insert_with(Vec::new).push(i);
        }

        // Return indices from cells that have at least min_cells points
        let mut candidates = Vec::new();
        for (_, indices) in cells.iter() {
            if indices.len() >= self.min_cells {
                candidates.extend(indices.iter().cloned());
            }
        }

        // If no cells meet the criteria, return all points
        if candidates.is_empty() {
            (0..n).collect()
        } else {
            candidates
        }
    }
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
