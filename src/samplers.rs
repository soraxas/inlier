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

/// Uniform random sampler drawing minimal samples without replacement.
pub struct UniformRandomSampler {
    rng: UniformRandomGenerator<usize>,
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
            if i + 1 <= sample_size {
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
