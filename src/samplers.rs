//! Sampling strategies for SupeRANSAC.
//!
//! This module will eventually contain multiple samplers (PROSAC, NAPSAC,
//! Progressive NAPSAC, etc.). We start with a simple uniform random sampler.

use crate::core::Sampler;
use crate::types::DataMatrix;
use crate::utils::UniformRandomGenerator;

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
    use super::{ProsacSampler, UniformRandomSampler};
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
}
