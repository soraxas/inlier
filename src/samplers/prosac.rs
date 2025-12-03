//! PROSAC sampler: progressively grows the subset of high-priority points.

use crate::core::Sampler;
use crate::types::DataMatrix;
use crate::utils::UniformRandomGenerator;

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

    pub fn initialize(&mut self, point_number: usize, sample_size: usize) {
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
            let t_n_plus1 = (i + 1) as f64 * t_n / (i + 1 - sample_size) as f64;
            self.growth_function[i] = t_n_prime + ((t_n_plus1 - t_n).ceil() as usize);
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
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
        let n = data.nrows();
        if sample_size == 0 || n == 0 || sample_size > n || out_indices.len() < sample_size {
            return false;
        }

        if self.sample_size.is_none() || self.point_number != n {
            self.initialize(n, sample_size);
        }

        if self.kth_sample_number > self.ransac_convergence_iterations {
            // Fall back to uniform random sampling
            self.rng
                .gen_unique(&mut out_indices[..sample_size], 0, n - 1);
        } else {
            // PROSAC sampling: sample from the current subset
            self.rng
                .gen_unique(&mut out_indices[..sample_size], 0, self.subset_size - 1);
        }

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
        // PROSAC updates are handled in increment_iteration_number
    }
}
