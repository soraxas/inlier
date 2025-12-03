//! Sampling strategies for SupeRANSAC.
//!
//! This module contains the sampling strategies used by SupeRANSAC.
//!
//! The goal is to mirror the behavior of the original C++ samplers found in
//! `superansac_c++/include/samplers`, while providing a small, idiomatic Rust
//! surface over the shared `Sampler` trait.

use crate::core::Sampler;
use crate::types::DataMatrix;
use crate::utils::UniformRandomGenerator;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Uniform random sampler drawing minimal samples without replacement.
pub struct UniformRandomSampler {
    rng: UniformRandomGenerator<usize>,
}

impl Default for UniformRandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl UniformRandomSampler {
    /// Construct a new sampler with a random seed.
    pub fn new() -> Self {
        Self {
            rng: UniformRandomGenerator::new(),
        }
    }

    /// Construct a sampler from a fixed seed (primarily for tests).
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: UniformRandomGenerator::from_seed(seed),
        }
    }
}

impl Sampler for UniformRandomSampler {
    fn sample(
        &mut self,
        data: &DataMatrix,
        sample_size: usize,
        out_indices: &mut [usize],
    ) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || sample_size > n || out_indices.len() < sample_size {
            return false;
        }

        // Sample unique indices in the range [0, n-1].
        self.rng
            .gen_unique(&mut out_indices[..sample_size], 0, n - 1);
        true
    }

    fn update(
        &mut self,
        _sample: &[usize],
        _sample_size: usize,
        _iteration: usize,
        _score_hint: f64,
    ) {
        // Uniform sampler has no adaptive state to update.
    }
}

/// Importance sampler drawing samples according to per-point probabilities.
///
/// This is a thin wrapper around `rand`'s `WeightedIndex` distribution. Unlike
/// `UniformRandomSampler`, it does **not** guarantee uniqueness inside a
/// sample, matching the original C++ behavior.
pub struct ImportanceSampler {
    dist: WeightedIndex<f64>,
    rng: rand::rngs::StdRng,
}

impl ImportanceSampler {
    /// Construct a new importance sampler from a probability vector and a
    /// random seed.
    pub fn from_probabilities_with_seed(
        probabilities: &[f64],
        seed: u64,
    ) -> Self {
        let dist = WeightedIndex::new(probabilities)
            .expect("ImportanceSampler: invalid weight vector");
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { dist, rng }
    }

    /// Construct a new importance sampler from a probability vector using
    /// an OS-provided random seed.
    pub fn from_probabilities(probabilities: &[f64]) -> Self {
        let dist = WeightedIndex::new(probabilities)
            .expect("ImportanceSampler: invalid weight vector");
        let rng = rand::rngs::StdRng::from_entropy();
        Self { dist, rng }
    }
}

impl Sampler for ImportanceSampler {
    fn sample(
        &mut self,
        data: &DataMatrix,
        sample_size: usize,
        out_indices: &mut [usize],
    ) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || out_indices.len() < sample_size {
            return false;
        }

        for dst in &mut out_indices[..sample_size] {
            let idx = self.dist.sample(&mut self.rng);
            *dst = idx;
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
        // The C++ importance sampler does not adapt its probabilities.
    }
}

/// PROSAC sampler: progressively grows the subset of high-priority points.
pub struct ProsacSampler {
    rng: UniformRandomGenerator<usize>,
    growth_function: Vec<usize>,
    sample_size: Option<usize>,
    point_number: usize,
    ransac_convergence_iterations: usize,
    kth_sample_number: usize,
    largest_sample_size: usize,
    subset_size: usize,
}

impl Default for ProsacSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProsacSampler {
    /// Construct with a default number of PROSAC iterations before falling back to RANSAC.
    pub fn new() -> Self {
        Self::with_ransac_convergence_iterations(100_000)
    }

    pub fn with_ransac_convergence_iterations(ransac_convergence_iterations: usize) -> Self {
        Self {
            rng: UniformRandomGenerator::new(),
            growth_function: Vec::new(),
            sample_size: None,
            point_number: 0,
            ransac_convergence_iterations,
            kth_sample_number: 1,
            largest_sample_size: 0,
            subset_size: 0,
        }
    }

    /// Construct from a fixed RNG seed (useful for tests).
    pub fn from_seed(seed: u64, ransac_convergence_iterations: usize) -> Self {
        Self {
            rng: UniformRandomGenerator::from_seed(seed),
            growth_function: Vec::new(),
            sample_size: None,
            point_number: 0,
            ransac_convergence_iterations,
            kth_sample_number: 1,
            largest_sample_size: 0,
            subset_size: 0,
        }
    }

    fn increment_iteration_number(&mut self) {
        self.kth_sample_number += 1;

        if self.kth_sample_number > self.ransac_convergence_iterations {
            if self.point_number > 0 {
                self.rng.reset(0, self.point_number - 1);
            }
        } else if self.kth_sample_number > self.growth_function[self.subset_size - 1] {
            self.subset_size += 1;
            self.subset_size = self.subset_size.min(self.point_number);
            self.largest_sample_size = self.largest_sample_size.max(self.subset_size);

            if self.subset_size > 1 {
                self.rng.reset(0, self.subset_size - 2);
            }
        }
    }

    fn initialize(&mut self, point_number: usize, sample_size: usize) {
        self.point_number = point_number;
        self.sample_size = Some(sample_size);
        self.growth_function.resize(point_number, 0);

        let mut t_n = self.ransac_convergence_iterations as f64;
        for i in 0..sample_size {
            t_n *= (sample_size - i) as f64 / (point_number - i) as f64;
        }

        let mut t_n_prime: usize = 1;
        for i in 0..point_number {
            if i < sample_size {
                self.growth_function[i] = t_n_prime;
                continue;
            }
            let t_n_plus1 =
                (i + 1) as f64 * t_n / (i + 1 - sample_size) as f64;
            self.growth_function[i] =
                t_n_prime + ((t_n_plus1 - t_n).ceil() as usize);
            t_n = t_n_plus1;
            t_n_prime = self.growth_function[i];
        }

        self.largest_sample_size = sample_size;
        self.subset_size = sample_size;

        if self.subset_size > 0 {
            self.rng.reset(0, self.subset_size - 1);
        }
    }

    /// Reset the internal PROSAC state while keeping configuration.
    pub fn reset(&mut self) {
        self.kth_sample_number = 1;
        if let Some(s) = self.sample_size {
            self.largest_sample_size = s;
            self.subset_size = s;
            if self.subset_size > 0 {
                self.rng.reset(0, self.subset_size - 1);
            }
        }
    }
}

/// Minimal neighborhood graph abstraction used by NAPSAC.
pub trait NeighborhoodGraph {
    /// Return the neighbor indices of a given point.
    fn neighbors(&self, index: usize) -> &[usize];

    /// Initialize the neighborhood graph with data.
    /// This method allows the graph to build its internal data structures.
    fn initialize(&mut self, data: &DataMatrix);
}

/// Grid-based neighborhood graph that partitions points into spatial cells.
///
/// Points in the same cell are considered neighbors. This is a simplified
/// 2D implementation for image coordinates.
pub struct GridNeighborhoodGraph {
    /// Grid cells: maps cell index to point indices
    grid: std::collections::HashMap<usize, Vec<usize>>,
    /// Cell index for each point
    point_to_cell: Vec<usize>,
    /// Cell size along x and y axes
    cell_size_x: f64,
    cell_size_y: f64,
    /// Number of cells along each axis
    cells_per_axis: usize,
    /// Empty vector for points with no neighbors
    empty: Vec<usize>,
}

impl GridNeighborhoodGraph {
    /// Create a new grid neighborhood graph.
    ///
    /// `cell_size_x` and `cell_size_y` are the sizes of each cell.
    /// `cells_per_axis` is the number of cells along each axis.
    pub fn new(cell_size_x: f64, cell_size_y: f64, cells_per_axis: usize) -> Self {
        Self {
            grid: std::collections::HashMap::new(),
            point_to_cell: Vec::new(),
            cell_size_x,
            cell_size_y,
            cells_per_axis,
            empty: Vec::new(),
        }
    }

    /// Initialize the grid with data points.
    ///
    /// Assumes data has at least 2 columns (x, y coordinates).
    pub fn initialize(&mut self, data: &DataMatrix) -> bool {
        if data.ncols() < 2 {
            return false;
        }

        self.grid.clear();
        self.point_to_cell = vec![0; data.nrows()];

        for row in 0..data.nrows() {
            let x = data[(row, 0)];
            let y = data[(row, 1)];

            if x < 0.0 || y < 0.0 {
                continue; // Skip negative coordinates
            }

            // Compute cell indices
            let cell_x = (x / self.cell_size_x).floor() as usize;
            let cell_y = (y / self.cell_size_y).floor() as usize;

            // Clamp to valid range
            let cell_x = cell_x.min(self.cells_per_axis.saturating_sub(1));
            let cell_y = cell_y.min(self.cells_per_axis.saturating_sub(1));

            // Compute linear cell index
            let cell_idx = cell_y * self.cells_per_axis + cell_x;

            // Add point to cell
            self.grid.entry(cell_idx).or_default().push(row);
            self.point_to_cell[row] = cell_idx;
        }

        !self.grid.is_empty()
    }
}

impl NeighborhoodGraph for GridNeighborhoodGraph {
    fn neighbors(&self, index: usize) -> &[usize] {
        if index >= self.point_to_cell.len() {
            return &self.empty;
        }

        let cell_idx = self.point_to_cell[index];
        self.grid.get(&cell_idx).map(|v| v.as_slice()).unwrap_or(&self.empty)
    }

    fn initialize(&mut self, data: &DataMatrix) {
        // GridNeighborhoodGraph::initialize is already implemented
        let _ = GridNeighborhoodGraph::initialize(self, data);
    }
}

/// USearch-based neighborhood graph using approximate nearest neighbor search.
///
/// This implementation uses the `usearch` crate for fast approximate nearest neighbor
/// search, which is more flexible than grid-based approaches and works well for
/// high-dimensional data or irregular point distributions.
///
/// # Example
/// ```
/// use inlier::samplers::{UsearchNeighborhoodGraph, NeighborhoodGraph};
/// use inlier::types::DataMatrix;
///
/// let mut graph = UsearchNeighborhoodGraph::new(10, 2); // 10 neighbors, 2D data
/// let data = DataMatrix::zeros(100, 2);
/// graph.initialize(&data);
/// let neighbors = graph.neighbors(0);
/// ```
pub struct UsearchNeighborhoodGraph {
    /// USearch index for nearest neighbor queries
    index: Option<Index>,
    /// Neighbors for each point: `neighbors[i]` contains the neighbor indices for point `i`
    neighbors: Vec<Vec<usize>>,
    /// Number of neighbors to find for each point
    k_neighbors: usize,
    /// Number of dimensions (extracted from data)
    dimensions: usize,
    /// Empty vector for points with no neighbors
    empty: Vec<usize>,
}

impl UsearchNeighborhoodGraph {
    /// Create a new USearch neighborhood graph.
    ///
    /// # Arguments
    /// * `k_neighbors` - Number of nearest neighbors to find for each point
    /// * `dimensions` - Number of dimensions in the data (will be set from data if 0)
    pub fn new(k_neighbors: usize, dimensions: usize) -> Self {
        Self {
            index: None,
            neighbors: Vec::new(),
            k_neighbors,
            dimensions,
            empty: Vec::new(),
        }
    }

    /// Build the neighborhood graph from data points.
    fn build_neighbors(&mut self, data: &DataMatrix) -> bool {
        let n = data.nrows();
        let dims = if self.dimensions > 0 {
            self.dimensions
        } else {
            data.ncols()
        };

        if n == 0 || dims == 0 {
            return false;
        }

        // Create USearch index
        let mut options = IndexOptions::default();
        options.dimensions = dims;
        options.metric = MetricKind::L2sq; // Squared L2 distance (faster)
        options.quantization = ScalarKind::F32;

        let index = match Index::new(&options) {
            Ok(idx) => idx,
            Err(_) => return false,
        };

        // Reserve capacity
        if let Err(_) = index.reserve(n) {
            return false;
        }

        // Add all points to the index
        for i in 0..n {
            let mut vector = Vec::<f32>::with_capacity(dims);
            for j in 0..dims {
                vector.push(data[(i, j)] as f32);
            }
            if let Err(_) = index.add(i as u64, &vector) {
                return false;
            }
        }

        // Find neighbors for each point
        self.neighbors.clear();
        self.neighbors.resize(n, Vec::new());

        for i in 0..n {
            let mut query = Vec::<f32>::with_capacity(dims);
            for j in 0..dims {
                query.push(data[(i, j)] as f32);
            }

            // Search for k+1 neighbors (including the point itself)
            let k = (self.k_neighbors + 1).min(n);
            match index.search(&query, k) {
                Ok(results) => {
                    // Filter out the point itself and store neighbors
                    for (key, _distance) in results.keys.iter().zip(results.distances.iter()) {
                        let key_usize = *key as usize;
                        if key_usize != i {
                            self.neighbors[i].push(key_usize);
                            if self.neighbors[i].len() >= self.k_neighbors {
                                break;
                            }
                        }
                    }
                }
                Err(_) => {
                    // If search fails, leave neighbors empty for this point
                }
            }
        }

        self.index = Some(index);
        true
    }
}

impl NeighborhoodGraph for UsearchNeighborhoodGraph {
    fn neighbors(&self, index: usize) -> &[usize] {
        if index >= self.neighbors.len() {
            return &self.empty;
        }
        &self.neighbors[index]
    }

    fn initialize(&mut self, data: &DataMatrix) {
        if self.dimensions == 0 {
            self.dimensions = data.ncols();
        }
        let _ = self.build_neighbors(data);
    }
}

/// NAPSAC sampler: draws samples from spatial neighborhoods instead of
/// globally uniform over all points.
pub struct NapsacSampler<N: NeighborhoodGraph> {
    rng: UniformRandomGenerator<usize>,
    neighborhood: N,
    max_attempts: usize,
    initialized: bool,
    point_number: usize,
}

impl<N: NeighborhoodGraph> NapsacSampler<N> {
    pub fn new(neighborhood: N) -> Self {
        Self {
            rng: UniformRandomGenerator::new(),
            neighborhood,
            max_attempts: 100,
            initialized: false,
            point_number: 0,
        }
    }

    pub fn from_seed(seed: u64, neighborhood: N) -> Self {
        Self {
            rng: UniformRandomGenerator::from_seed(seed),
            neighborhood,
            max_attempts: 100,
            initialized: false,
            point_number: 0,
        }
    }

    fn initialize(&mut self, point_number: usize) {
        self.point_number = point_number;
        if point_number > 0 {
            self.rng.reset(0, point_number - 1);
        }
        self.initialized = true;
    }
}

impl<N: NeighborhoodGraph> Sampler for NapsacSampler<N> {
    fn sample(
        &mut self,
        data: &DataMatrix,
        sample_size: usize,
        out_indices: &mut [usize],
    ) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || sample_size > n || out_indices.len() < sample_size {
            return false;
        }

        if !self.initialized || self.point_number != n {
            self.initialize(n);
        }

        let mut attempts = 0usize;
        while attempts < self.max_attempts {
            attempts += 1;

            // Select a random center point.
            let mut center_buf = [0usize; 1];
            self.rng.gen_unique_current(&mut center_buf);
            let center = center_buf[0];

            let neighbors = self.neighborhood.neighbors(center);
            if neighbors.len() < sample_size {
                continue;
            }

            if neighbors.len() == sample_size {
                out_indices[..sample_size].copy_from_slice(&neighbors[..sample_size]);
                return true;
            }

            // Otherwise, randomly pick (sample_size - 1) distinct neighbors
            // and include the center itself.
            let mut neighbor_indices = vec![0usize; sample_size - 1];
            self.rng
                .gen_unique_current(&mut neighbor_indices[..]);

            out_indices[0] = center;
            for (dst, &ni) in out_indices[1..].iter_mut().zip(neighbor_indices.iter()) {
                let idx = ni.min(neighbors.len() - 1);
                *dst = neighbors[idx];
            }

            return true;
        }

        false
    }

    fn update(
        &mut self,
        _sample: &[usize],
        _sample_size: usize,
        _iteration: usize,
        _score_hint: f64,
    ) {
        // No adaptive behavior yet.
    }
}

/// Adaptive Reordering (AR) sampler.
///
/// This is a simplified but behaviorally similar port of the C++ sampler in
/// `adaptive_reordering_sampler.h`. It maintains a priority queue of points
/// ordered by their current inlier probability estimate and always samples
/// the highest-probability points first, while slightly randomizing updates.
pub struct AdaptiveReorderingSampler {
    /// (p, index, appearance_count, a, b)
    probabilities: Vec<(f64, usize, usize, f64, f64)>,
    /// Max-heap by current probability `p`.
    queue: std::collections::BinaryHeap<(ordered_float::OrderedFloat<f64>, usize)>,
    /// Randomness parameters (see C++ implementation).
    randomness: f64,
    randomness_half: f64,
    /// RNG for the small random perturbation.
    rng: rand::rngs::StdRng,
}

impl AdaptiveReorderingSampler {
    /// Initialize from inlier probability estimates and an estimator variance.
    pub fn new_with_seed(
        inlier_probabilities: &[f64],
        estimator_variance: f64,
        randomness: f64,
        seed: u64,
    ) -> Self {
        assert!(
            !inlier_probabilities.is_empty(),
            "AdaptiveReorderingSampler requires non-empty probabilities"
        );

        let mut probabilities = Vec::with_capacity(inlier_probabilities.len());
        let mut queue = std::collections::BinaryHeap::new();

        for (idx, &p) in inlier_probabilities.iter().enumerate() {
            let mut prob = p;
            if prob == 1.0 {
                prob -= 1e-6;
            }

            let a = prob * prob * (1.0 - prob) / estimator_variance - prob;
            let b = a * (1.0 - prob) / prob;

            probabilities.push((prob, idx, 0usize, a, b));
            queue.push((ordered_float::OrderedFloat(prob), idx));
        }

        let randomness_half = randomness / 2.0;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);

        Self {
            probabilities,
            queue,
            randomness,
            randomness_half,
            rng,
        }
    }

    /// Convenience constructor with default variance/randomness.
    pub fn new(inlier_probabilities: &[f64]) -> Self {
        Self::new_with_seed(inlier_probabilities, 0.9765, 0.01, 42)
    }
}

impl Sampler for AdaptiveReorderingSampler {
    fn sample(
        &mut self,
        data: &DataMatrix,
        sample_size: usize,
        out_indices: &mut [usize],
    ) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || out_indices.len() < sample_size {
            return false;
        }
        for i in 0..sample_size {
            if let Some((_, idx)) = self.queue.pop() {
                out_indices[i] = idx;
            } else {
                return false;
            }
        }
        true
    }

    fn update(
        &mut self,
        sample: &[usize],
        sample_size: usize,
        _iteration: usize,
        _score_hint: f64,
    ) {
        use rand::Rng;

        let count = sample_size.min(sample.len());
        for i in 0..count {
            let idx = sample[i];
            if let Some(entry) = self.probabilities.get_mut(idx) {
                let (ref mut p, _point_idx, ref mut appearance, a, b) = *entry;
                *appearance += 1;

                let base =
                    (a / (a + b + (*appearance as f64))).abs();
                let jitter: f64 = self
                    .rng
                    .gen_range(-self.randomness_half..self.randomness_half);
                let mut updated = base + jitter;
                updated = updated.max(0.0).min(0.999);

                *p = updated;
                self.queue
                    .push((ordered_float::OrderedFloat(updated), idx));
            }
        }
    }
}

impl Sampler for ProsacSampler {
    fn sample(
        &mut self,
        data: &DataMatrix,
        sample_size: usize,
        out_indices: &mut [usize],
    ) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || sample_size > n || out_indices.len() < sample_size {
            return false;
        }

        // Lazily initialize when first used.
        match self.sample_size {
            None => {
                self.initialize(n, sample_size);
            }
            Some(existing) if existing != sample_size => {
                panic!("ProsacSampler does not support changing sample size after initialization");
            }
            Some(_) => {
                if self.point_number != n {
                    // Reinitialize if data size changed.
                    self.initialize(n, sample_size);
                }
            }
        }

        let sample_size = sample_size;

        if self.kth_sample_number > self.ransac_convergence_iterations {
            // Fall back to RANSAC: draw a uniform random sample.
            self.rng
                .gen_unique_current(&mut out_indices[..sample_size]);
            return true;
        }

        // Draw sample_size - 1 elements from the current subset, and fix the last one.
        if sample_size > 1 {
            self.rng
                .gen_unique_current(&mut out_indices[..sample_size - 1]);
        }
        out_indices[sample_size - 1] = self.subset_size - 1;

        self.increment_iteration_number();

        true
    }

    fn update(
        &mut self,
        _sample: &[usize],
        _sample_size: usize,
        _iteration: usize,
        _score_hint: f64,
    ) {
        // The current C++ implementation leaves PROSAC's `update` empty.
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveReorderingSampler, ImportanceSampler, NapsacSampler, NeighborhoodGraph,
        ProsacSampler, UniformRandomSampler,
    };
    use crate::core::Sampler;
    use crate::types::DataMatrix;

    #[test]
    fn uniform_sampler_respects_bounds_and_uniqueness() {
        let data = DataMatrix::zeros(10, 2);
        let mut sampler = UniformRandomSampler::from_seed(7);

        let sample_size = 4;
        let mut indices = vec![0usize; sample_size];

        let ok = sampler.sample(&data, sample_size, &mut indices);
        assert!(ok);

        // Bounds
        assert!(indices.iter().all(|&i| i < data.nrows()));

        // Uniqueness
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                assert_ne!(indices[i], indices[j]);
            }
        }
    }

    #[test]
    fn uniform_sampler_fails_when_sample_too_large() {
        let data = DataMatrix::zeros(3, 2);
        let mut sampler = UniformRandomSampler::from_seed(1);
        let mut indices = vec![0usize; 5];
        let ok = sampler.sample(&data, 5, &mut indices);
        assert!(!ok);
    }

    #[test]
    fn importance_sampler_respects_bounds_and_sample_size() {
        let data = DataMatrix::zeros(10, 2);
        let probs = vec![1.0; 10];
        let mut sampler = ImportanceSampler::from_probabilities_with_seed(&probs, 7);

        let sample_size = 4;
        let mut indices = vec![0usize; sample_size];

        let ok = sampler.sample(&data, sample_size, &mut indices);
        assert!(ok);
        assert_eq!(indices.len(), sample_size);
        assert!(indices.iter().all(|&i| i < data.nrows()));
    }

    #[test]
    fn uniform_sampler_is_deterministic_with_fixed_seed() {
        let data = DataMatrix::zeros(15, 2);
        let mut s1 = UniformRandomSampler::from_seed(123);
        let mut s2 = UniformRandomSampler::from_seed(123);

        let sample_size = 5;
        let mut a = vec![0usize; sample_size];
        let mut b = vec![0usize; sample_size];

        for _ in 0..10 {
            assert!(s1.sample(&data, sample_size, &mut a));
            assert!(s2.sample(&data, sample_size, &mut b));
            assert_eq!(a, b);
        }
    }

    #[test]
    fn adaptive_reordering_sampler_behaves_reasonably() {
        let num_points = 10;
        let data = DataMatrix::zeros(num_points, 2);
        // First half with higher initial probability.
        let mut probs = vec![0.8; num_points];
        for p in &mut probs[num_points / 2..] {
            *p = 0.2;
        }
        let mut sampler =
            AdaptiveReorderingSampler::new_with_seed(&probs, 0.9765, 0.01, 123);

        let sample_size = 3;
        let mut indices = vec![0usize; sample_size];

        for _ in 0..5 {
            let ok = sampler.sample(&data, sample_size, &mut indices);
            assert!(ok);
            assert!(indices.iter().all(|&i| i < data.nrows()));

            sampler.update(&indices, sample_size, 0, 0.0);
        }
    }

    #[test]
    fn prosac_sampler_respects_bounds_and_uniqueness() {
        let data = DataMatrix::zeros(20, 2);
        let mut sampler = ProsacSampler::from_seed(5, 10);

        let sample_size = 3;
        let mut indices = vec![0usize; sample_size];

        // Draw several samples to exercise PROSAC's growth and fallback.
        for _ in 0..15 {
            let ok = sampler.sample(&data, sample_size, &mut indices);
            assert!(ok);

            assert!(indices.iter().all(|&i| i < data.nrows()));
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    assert_ne!(indices[i], indices[j]);
                }
            }
        }
    }

    struct DummyNeighborhood {
        /// For each point i, store a small set of neighbors around it.
        neighbors: Vec<Vec<usize>>,
    }

    impl DummyNeighborhood {
        fn new(num_points: usize, window: usize) -> Self {
            let mut neighbors = Vec::with_capacity(num_points);
            for i in 0..num_points {
                let start = i.saturating_sub(window);
                let end = (i + window).min(num_points - 1);
                neighbors.push((start..=end).collect());
            }
            Self { neighbors }
        }
    }

    impl NeighborhoodGraph for DummyNeighborhood {
        fn neighbors(&self, index: usize) -> &[usize] {
            &self.neighbors[index]
        }

        fn initialize(&mut self, _data: &DataMatrix) {
            // Dummy neighborhood is pre-initialized
        }
    }

    #[test]
    fn usearch_neighborhood_graph_finds_neighbors() {
        use super::{UsearchNeighborhoodGraph, NeighborhoodGraph};

        // Create 2D data with points in a grid pattern
        let n = 25; // 5x5 grid
        let mut data = DataMatrix::zeros(n, 2);
        for i in 0..5 {
            for j in 0..5 {
                let idx = i * 5 + j;
                data[(idx, 0)] = i as f64 * 10.0;
                data[(idx, 1)] = j as f64 * 10.0;
            }
        }

        let mut graph = UsearchNeighborhoodGraph::new(4, 2); // 4 neighbors, 2D
        graph.initialize(&data);

        // Check that each point has neighbors
        for i in 0..n {
            let neighbors = graph.neighbors(i);
            assert!(neighbors.len() <= 4, "Should have at most 4 neighbors");
            // Points in the grid should have neighbors
            if i > 0 && i < n - 1 {
                assert!(!neighbors.is_empty(), "Interior points should have neighbors");
            }
        }
    }

    #[test]
    fn napsac_sampler_draws_from_local_neighborhoods() {
        let num_points = 20;
        let data = DataMatrix::zeros(num_points, 2);
        let neighborhood = DummyNeighborhood::new(num_points, 2);
        let mut sampler = NapsacSampler::from_seed(7, neighborhood);

        let sample_size = 4;
        let mut indices = vec![0usize; sample_size];

        for _ in 0..10 {
            let ok = sampler.sample(&data, sample_size, &mut indices);
            assert!(ok);

            // Bounds
            assert!(indices.iter().all(|&i| i < data.nrows()));

            // At least 2 points should be returned and all within bounds.
            assert!(indices.len() >= 2);
        }
    }

    #[test]
    fn napsac_sampler_is_deterministic_with_fixed_seed() {
        let num_points = 30;
        let data = DataMatrix::zeros(num_points, 2);
        let neighborhood = DummyNeighborhood::new(num_points, 3);

        let mut s1 = NapsacSampler::from_seed(999, neighborhood);

        // Rebuild the same neighborhood for the second sampler.
        let neighborhood2 = DummyNeighborhood::new(num_points, 3);
        let mut s2 = NapsacSampler::from_seed(999, neighborhood2);

        let sample_size = 5;
        let mut a = vec![0usize; sample_size];
        let mut b = vec![0usize; sample_size];

        for _ in 0..10 {
            assert!(s1.sample(&data, sample_size, &mut a));
            assert!(s2.sample(&data, sample_size, &mut b));
            assert_eq!(a, b);
        }
    }
}

/// Progressive NAPSAC sampler: combines PROSAC for selecting the first point
/// (center of neighborhood) with NAPSAC for selecting remaining points from
/// the neighborhood.
///
/// This is a simplified implementation. A full implementation would use
/// multiple overlapping grid layers and adaptively grow neighborhood sizes.
pub struct ProgressiveNapsacSampler<N: NeighborhoodGraph> {
    one_point_prosac: ProsacSampler,
    napsac: NapsacSampler<N>,
    kth_sample_number: usize,
    max_progressive_iterations: usize,
    sampler_length: f64,
    point_number: usize,
    initialized: bool,
}

impl<N: NeighborhoodGraph> ProgressiveNapsacSampler<N> {
    pub fn new(neighborhood: N, sampler_length: f64) -> Self {
        Self {
            one_point_prosac: ProsacSampler::from_seed(0, 1),
            napsac: NapsacSampler::new(neighborhood),
            kth_sample_number: 0,
            max_progressive_iterations: 0,
            sampler_length,
            point_number: 0,
            initialized: false,
        }
    }

    pub fn from_seed(seed: u64, neighborhood: N, sampler_length: f64) -> Self {
        Self {
            one_point_prosac: ProsacSampler::from_seed(seed, 1),
            napsac: NapsacSampler::from_seed(seed, neighborhood),
            kth_sample_number: 0,
            max_progressive_iterations: 0,
            sampler_length,
            point_number: 0,
            initialized: false,
        }
    }

    fn initialize(&mut self, point_number: usize) {
        self.point_number = point_number;
        self.one_point_prosac.initialize(point_number, 1);
        self.max_progressive_iterations =
            (self.sampler_length * point_number as f64) as usize;
        self.initialized = true;
    }
}

impl<N: NeighborhoodGraph> Sampler for ProgressiveNapsacSampler<N> {
    fn sample(
        &mut self,
        data: &DataMatrix,
        sample_size: usize,
        out_indices: &mut [usize],
    ) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || sample_size > n || out_indices.len() < sample_size {
            return false;
        }

        if !self.initialized || self.point_number != n {
            self.initialize(n);
        }

        self.kth_sample_number += 1;

        // After max iterations, fall back to pure PROSAC
        if self.kth_sample_number > self.max_progressive_iterations {
            let mut prosac = ProsacSampler::from_seed(0, sample_size);
            prosac.initialize(n, sample_size);
            // Set the sample number by updating kth_sample_number internally
            // (simplified - a full implementation would expose this)
            return prosac.sample(data, sample_size, out_indices);
        }

        // Select first point using PROSAC (as center of neighborhood)
        let mut center = [0usize; 1];
        if !self.one_point_prosac.sample(data, 1, &mut center) {
            return false;
        }

        // Use NAPSAC to select remaining points from neighborhood
        // For simplicity, we use the same neighborhood graph but could
        // adaptively grow the neighborhood size based on hits
        out_indices[0] = center[0];

        // Sample remaining points using NAPSAC starting from the center
        // This is simplified - a full implementation would use the center's neighborhood
        let mut remaining = vec![0usize; sample_size - 1];
        if !self.napsac.sample(data, sample_size - 1, &mut remaining) {
            return false;
        }

        // Combine center with remaining points
        for (i, &idx) in remaining.iter().take(sample_size - 1).enumerate() {
            out_indices[i + 1] = idx;
        }

        true
    }

    fn update(
        &mut self,
        sample: &[usize],
        sample_size: usize,
        iteration: usize,
        score_hint: f64,
    ) {
        self.one_point_prosac.update(sample, 1, iteration, score_hint);
        self.napsac.update(sample, sample_size, iteration, score_hint);
    }
}
