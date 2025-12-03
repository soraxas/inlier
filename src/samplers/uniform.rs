//! Uniform random sampler drawing minimal samples without replacement.

use crate::core::Sampler;
use crate::types::DataMatrix;
use crate::utils::UniformRandomGenerator;

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
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
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
