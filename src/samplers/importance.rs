//! Importance sampler drawing samples according to per-point probabilities.

use crate::core::Sampler;
use crate::types::DataMatrix;
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};

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
    pub fn from_probabilities_with_seed(probabilities: &[f64], seed: u64) -> Self {
        let dist =
            WeightedIndex::new(probabilities).expect("ImportanceSampler: invalid weight vector");
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { dist, rng }
    }

    /// Construct a new importance sampler from a probability vector using
    /// an OS-provided random seed.
    pub fn from_probabilities(probabilities: &[f64]) -> Self {
        let dist =
            WeightedIndex::new(probabilities).expect("ImportanceSampler: invalid weight vector");
        let rng = rand::rngs::StdRng::from_entropy();
        Self { dist, rng }
    }
}

impl Sampler for ImportanceSampler {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
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
