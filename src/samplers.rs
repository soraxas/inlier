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

#[cfg(test)]
mod tests {
    use super::UniformRandomSampler;
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
}
