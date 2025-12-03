//! Progressive NAPSAC sampler: combines PROSAC with NAPSAC.

use crate::core::Sampler;
use crate::samplers::neighborhood::NeighborhoodGraph;
use crate::samplers::napsac::NapsacSampler;
use crate::samplers::prosac::ProsacSampler;
use crate::types::DataMatrix;

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
        self.max_progressive_iterations = (self.sampler_length * point_number as f64) as usize;
        self.initialized = true;
    }
}

impl<N: NeighborhoodGraph> Sampler for ProgressiveNapsacSampler<N> {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
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

    fn update(&mut self, sample: &[usize], sample_size: usize, iteration: usize, score_hint: f64) {
        self.one_point_prosac
            .update(sample, 1, iteration, score_hint);
        self.napsac
            .update(sample, sample_size, iteration, score_hint);
    }
}
