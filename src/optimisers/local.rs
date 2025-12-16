use crate::core::{Estimator, Sampler};
use crate::samplers::UniformRandomSampler;
use crate::types::DataMatrix;

#[cfg(feature = "graph-cut")]
use petgraph::graph::UnGraph;

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

impl<E, S> super::LocalOptimizer<E::Model, S> for LeastSquaresOptimizer<E>
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

        let refined_models = self
            .estimator
            .estimate_model_nonminimal(data, inliers, None);

        if refined_models.is_empty() {
            (model.clone(), best_score.clone(), inliers.to_vec())
        } else {
            (
                refined_models[0].clone(),
                best_score.clone(),
                inliers.to_vec(),
            )
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

impl<E, S> super::LocalOptimizer<E::Model, S> for IteratedLeastSquaresOptimizer<E>
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

        let mut current_model = model.clone();
        let current_inliers = inliers.to_vec();

        for _iteration in 0..self.max_iterations {
            let refined_models =
                self.estimator
                    .estimate_model_nonminimal(data, &current_inliers, None);

            if refined_models.is_empty() {
                break;
            }

            current_model = refined_models[0].clone();
        }

        (current_model, best_score.clone(), current_inliers)
    }
}

/// Nested RANSAC optimizer that runs an inner RANSAC loop on the inliers.
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

impl<E, S> super::LocalOptimizer<E::Model, S> for NestedRansacOptimizer<E>
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
        if inliers.len() < self.estimator.sample_size() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        let mut current_model = model.clone();
        let current_inliers = inliers.to_vec();
        let current_score = best_score.clone();

        let non_minimal_sample_size = self.sample_size_multiplier * self.estimator.sample_size();
        let mut sampler = UniformRandomSampler::new();

        for _iteration in 0..self.max_iterations {
            let mut current_sample_size = current_inliers.len().saturating_sub(1);
            if current_sample_size >= non_minimal_sample_size {
                current_sample_size = non_minimal_sample_size;
            }

            if current_sample_size < self.estimator.sample_size() {
                break;
            }

            let mut sample = vec![0usize; current_sample_size];
            if current_sample_size == current_inliers.len() {
                sample.copy_from_slice(&current_inliers[..current_sample_size]);
            } else {
                let mut temp_indices = vec![0usize; current_sample_size];
                let dummy_data = DataMatrix::zeros(current_inliers.len(), data.ncols());
                if !sampler.sample(&dummy_data, current_sample_size, &mut temp_indices) {
                    continue;
                }
                for (i, &temp_idx) in temp_indices.iter().enumerate() {
                    sample[i] = current_inliers[temp_idx];
                }
            }

            let refined_models = self
                .estimator
                .estimate_model_nonminimal(data, &sample, None);
            if refined_models.is_empty() {
                continue;
            }

            current_model = refined_models[0].clone();
        }

        (current_model, current_score, current_inliers)
    }
}

/// Iteratively Reweighted Least Squares (IRLS) optimizer.
pub struct IRLSOptimizer<E, F>
where
    E: Estimator,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
{
    estimator: E,
    residual_fn: F,
    threshold: f64,
    max_iterations: usize,
    convergence_threshold: f64,
    use_inliers: bool,
}

impl<E, F> IRLSOptimizer<E, F>
where
    E: Estimator,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
{
    pub fn new(estimator: E, residual_fn: F, threshold: f64) -> Self {
        Self {
            estimator,
            residual_fn,
            threshold,
            max_iterations: 100,
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

    fn compute_weights(&self, data: &DataMatrix, model: &E::Model, indices: &[usize]) -> Vec<f64> {
        let thresh_sq = self.threshold * self.threshold;
        let mut weights = Vec::with_capacity(indices.len());

        for &idx in indices {
            let r = (self.residual_fn)(data, model, idx);
            let r_sq = r * r;

            if r_sq < thresh_sq {
                weights.push(1.0 - r_sq / thresh_sq);
            } else {
                weights.push(0.0);
            }
        }

        weights
    }
}

impl<E, F, S> super::LocalOptimizer<E::Model, S> for IRLSOptimizer<E, F>
where
    E: Estimator,
    E::Model: Clone,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
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

        let mut current_model = model.clone();

        for _iteration in 0..self.max_iterations {
            let weights = self.compute_weights(data, &current_model, inliers);
            if weights.iter().copied().sum::<f64>() < self.convergence_threshold {
                break;
            }

            let refined_models =
                self.estimator
                    .estimate_model_nonminimal(data, inliers, Some(&weights));
            if refined_models.is_empty() {
                break;
            }

            current_model = refined_models[0].clone();
        }

        (current_model, best_score.clone(), inliers.to_vec())
    }
}

#[cfg(feature = "graph-cut")]
pub struct GraphCutLocalOptimizer<E, F>
where
    E: Estimator,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
{
    estimator: E,
    residual_fn: F,
    threshold: f64,
    k_neighbors: usize,
    smooth_weight: f64,
    max_iterations: usize,
}

#[cfg(feature = "graph-cut")]
impl<E, F> GraphCutLocalOptimizer<E, F>
where
    E: Estimator,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
{
    pub fn new(estimator: E, residual_fn: F, threshold: f64) -> Self {
        Self {
            estimator,
            residual_fn,
            threshold,
            k_neighbors: 8,
            smooth_weight: 0.5,
            max_iterations: 5,
        }
    }

    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.max(1);
        self
    }

    pub fn with_smooth_weight(mut self, weight: f64) -> Self {
        self.smooth_weight = weight.max(0.0);
        self
    }

    fn build_graph(&self, data: &DataMatrix) -> UnGraph<(), f64> {
        let n = data.nrows();
        let mut graph: UnGraph<(), f64> = UnGraph::default();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let diff = data.row(i) - data.row(j);
                    let dist = diff.norm();
                    (j, dist)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, dist) in dists.iter().take(self.k_neighbors.min(n.saturating_sub(1))) {
                graph.update_edge(nodes[i], nodes[j], dist);
            }
        }

        graph
    }
}

#[cfg(feature = "graph-cut")]
impl<E, F, S> super::LocalOptimizer<E::Model, S> for GraphCutLocalOptimizer<E, F>
where
    E: Estimator,
    E::Model: Clone,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
    S: Clone,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &E::Model,
        best_score: &S,
    ) -> (E::Model, S, Vec<usize>) {
        let n = data.nrows();
        if n == 0 || inliers.len() < self.estimator.sample_size() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        let graph = self.build_graph(data);
        let mut labels = vec![false; n];
        for &idx in inliers {
            if idx < n {
                labels[idx] = true;
            }
        }

        for _ in 0..self.max_iterations {
            let mut changed = false;
            let mut new_labels = labels.clone();

            for node in graph.node_indices() {
                let idx = node.index();
                let residual = (self.residual_fn)(data, model, idx).abs();
                let unary_in = residual / self.threshold.max(1e-9);
                let unary_out = 1.0;

                let mut smooth_in = 0.0;
                let mut smooth_out = 0.0;
                for neighbor in graph.neighbors(node) {
                    let n_label = labels[neighbor.index()];
                    if n_label {
                        smooth_out += self.smooth_weight;
                    } else {
                        smooth_in += self.smooth_weight;
                    }
                }

                let e_in = unary_in + smooth_in;
                let e_out = unary_out + smooth_out;
                new_labels[idx] = e_in <= e_out;
                if new_labels[idx] != labels[idx] {
                    changed = true;
                }
            }

            labels = new_labels;
            if !changed {
                break;
            }
        }

        let refined_inliers: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| if l { Some(i) } else { None })
            .collect();

        if refined_inliers.len() < self.estimator.sample_size() {
            return (model.clone(), best_score.clone(), refined_inliers);
        }

        let refined_models = self
            .estimator
            .estimate_model_nonminimal(data, &refined_inliers, None);

        if let Some(refined) = refined_models.first() {
            return (refined.clone(), best_score.clone(), refined_inliers);
        }

        (model.clone(), best_score.clone(), refined_inliers)
    }
}

/// Cross-validation optimizer that uses bootstrap sampling to compute point weights.
pub struct CrossValidationOptimizer<E, F>
where
    E: Estimator,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
{
    estimator: E,
    residual_fn: F,
    threshold: f64,
    repetitions: usize,
    sample_size_multiplier: f64,
    use_inliers: bool,
}

impl<E, F> CrossValidationOptimizer<E, F>
where
    E: Estimator,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
{
    pub fn new(estimator: E, residual_fn: F, threshold: f64) -> Self {
        Self {
            estimator,
            residual_fn,
            threshold,
            repetitions: 100,
            sample_size_multiplier: 0.5,
            use_inliers: false,
        }
    }

    pub fn with_repetitions(mut self, repetitions: usize) -> Self {
        self.repetitions = repetitions;
        self
    }

    pub fn with_sample_size_multiplier(mut self, multiplier: f64) -> Self {
        self.sample_size_multiplier = multiplier;
        self
    }

    pub fn set_use_inliers(&mut self, use_inliers: bool) {
        self.use_inliers = use_inliers;
    }
}

impl<E, F, S> super::LocalOptimizer<E::Model, S> for CrossValidationOptimizer<E, F>
where
    E: Estimator,
    E::Model: Clone,
    F: Fn(&DataMatrix, &E::Model, usize) -> f64,
    S: Clone,
{
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &E::Model,
        best_score: &S,
    ) -> (E::Model, S, Vec<usize>) {
        let inlier_count = inliers.len();
        let minimal_sample_size = self.estimator.sample_size();

        if inlier_count < minimal_sample_size {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        let sample_size = (self.sample_size_multiplier * inlier_count as f64)
            .max(minimal_sample_size as f64) as usize;

        let mut accumulated_scores = vec![0.0; inlier_count];
        let mut rng = rand::rng();

        for _rep in 0..self.repetitions {
            let mut bootstrap_sample = Vec::with_capacity(sample_size);
            for _ in 0..sample_size {
                let idx = rand::Rng::random_range(&mut rng, 0..inlier_count);
                bootstrap_sample.push(inliers[idx]);
            }

            let bootstrap_models = self.estimator.estimate_model(data, &bootstrap_sample);
            if bootstrap_models.is_empty() {
                continue;
            }

            let bootstrap_model = &bootstrap_models[0];

            for (i, &inlier_idx) in inliers.iter().enumerate() {
                let error = (self.residual_fn)(data, bootstrap_model, inlier_idx);
                let score = (1.0 - error / self.threshold).max(0.0);
                accumulated_scores[i] += score;
            }
        }

        let weights: Vec<f64> = accumulated_scores
            .iter()
            .map(|&score| score / self.repetitions as f64)
            .collect();

        let refined_models =
            self.estimator
                .estimate_model_nonminimal(data, inliers, Some(&weights));
        if refined_models.is_empty() {
            return (model.clone(), best_score.clone(), inliers.to_vec());
        }

        (
            refined_models[0].clone(),
            best_score.clone(),
            inliers.to_vec(),
        )
    }
}
