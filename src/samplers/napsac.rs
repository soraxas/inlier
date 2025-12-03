//! NAPSAC sampler: draws samples from spatial neighborhoods.

use crate::core::Sampler;
use crate::samplers::neighborhood::NeighborhoodGraph;
use crate::types::DataMatrix;
use crate::utils::UniformRandomGenerator;

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
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
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
            self.rng.gen_unique_current(&mut neighbor_indices[..]);

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
