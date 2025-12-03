//! Adaptive Reordering (AR) sampler.

use crate::core::Sampler;
use crate::types::DataMatrix;
use rand::Rng;
use rand::SeedableRng;

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
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
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
        let count = sample_size.min(sample.len());
        for i in 0..count {
            let idx = sample[i];
            if let Some(entry) = self.probabilities.get_mut(idx) {
                let (ref mut p, _point_idx, ref mut appearance, a, b) = *entry;
                *appearance += 1;

                let base = (a / (a + b + (*appearance as f64))).abs();
                let jitter: f64 = self
                    .rng
                    .gen_range(-self.randomness_half..self.randomness_half);
                let mut updated = base + jitter;
                updated = updated.max(0.0).min(0.999);

                *p = updated;
                self.queue.push((ordered_float::OrderedFloat(updated), idx));
            }
        }
    }
}
