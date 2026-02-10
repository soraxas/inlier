//! Truncated Least Squares (TLS) estimator
//!
//! Port of TEASER++'s ScalarTLSEstimator for robust parameter estimation.
//! Reference: TEASER-plusplus/teaser/src/registration.cc

/// TLS estimator for scalar parameter estimation with outlier rejection
///
/// Given measurements X and their uncertainty ranges, estimates the true
/// parameter value using weighted least squares with truncation.
pub struct ScalarTLSEstimator;

impl ScalarTLSEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Estimate scalar parameter using Truncated Least Squares
    ///
    /// # Arguments
    /// * `measurements` - Observed values (e.g., scale ratios)
    /// * `ranges` - Maximum admissible error for each measurement
    ///
    /// # Returns
    /// * Estimated parameter value and inlier mask
    ///
    /// # Algorithm
    /// 1. Create intervals [X_i - range_i, X_i + range_i]
    /// 2. Sort all interval boundaries
    /// 3. For each boundary, compute weighted LS estimate
    /// 4. Select estimate that minimizes cost function
    pub fn estimate(&self, measurements: &[f64], ranges: &[f64]) -> Option<(f64, Vec<bool>)> {
        let n = measurements.len();

        if n == 0 || measurements.len() != ranges.len() {
            return None;
        }

        if n == 1 {
            return Some((measurements[0], vec![true]));
        }

        // Create interval boundaries with indices
        // Positive index = interval start, negative = interval end
        let mut boundaries = Vec::with_capacity(2 * n);
        for i in 0..n {
            boundaries.push((measurements[i] - ranges[i], (i + 1) as i32));
            boundaries.push((measurements[i] + ranges[i], -((i + 1) as i32)));
        }

        // Sort boundaries in ascending order
        boundaries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Compute weights (inverse squared ranges)
        let weights: Vec<f64> = ranges.iter().map(|r| 1.0 / (r * r)).collect();

        let nr_centers = 2 * n;
        let mut x_hat = vec![0.0; nr_centers];
        let mut x_cost = vec![0.0; nr_centers];

        // Initialize accumulators
        let mut ranges_inverse_sum: f64 = ranges.iter().sum();
        let mut dot_x_weights = 0.0;
        let mut dot_weights_consensus = 0.0;
        let mut consensus_set_cardinal = 0;
        let mut sum_xi = 0.0;
        let mut sum_xi_square = 0.0;

        // Sweep through boundaries, computing estimates
        for i in 0..nr_centers {
            let idx = (boundaries[i].1.abs() - 1) as usize;
            let epsilon = if boundaries[i].1 > 0 { 1.0 } else { -1.0 };

            // Update consensus set statistics
            consensus_set_cardinal += epsilon as i32;
            dot_weights_consensus += epsilon * weights[idx];
            dot_x_weights += epsilon * weights[idx] * measurements[idx];
            ranges_inverse_sum -= epsilon * ranges[idx];
            sum_xi += epsilon * measurements[idx];
            sum_xi_square += epsilon * measurements[idx] * measurements[idx];

            // Weighted least squares estimate
            x_hat[i] = dot_x_weights / dot_weights_consensus;

            // Cost function: residual + penalty for unused ranges
            let residual = (consensus_set_cardinal as f64) * x_hat[i] * x_hat[i] + sum_xi_square
                - 2.0 * sum_xi * x_hat[i];
            x_cost[i] = residual + ranges_inverse_sum;
        }

        // Find estimate with minimum cost
        let min_idx = x_cost
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)?;

        let estimate = x_hat[min_idx];

        // Determine inliers: |X_i - estimate| <= range_i
        let inliers = measurements
            .iter()
            .zip(ranges.iter())
            .map(|(x, r)| (x - estimate).abs() <= *r)
            .collect();

        Some((estimate, inliers))
    }
}

impl Default for ScalarTLSEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_no_outliers() {
        // Test from TEASER++ tls-test.cc
        let estimator = ScalarTLSEstimator::new();

        let measurements = vec![0.5, 1.0, 0.6, 0.7, 1.2];
        let ranges = vec![0.9, 0.9, 0.4, 0.5, 0.4];

        let result = estimator.estimate(&measurements, &ranges);
        assert!(result.is_some());

        let (estimate, inliers) = result.unwrap();

        // Reference from TEASER++: 0.8383
        assert!((estimate - 0.8383).abs() < 0.001);

        // All should be inliers
        assert_eq!(inliers.len(), 5);
        assert!(inliers.iter().all(|&x| x));
    }

    #[test]
    fn test_tls_one_outlier() {
        let estimator = ScalarTLSEstimator::new();

        let measurements = vec![0.5, 1.0, 0.6, 0.7, 1.2, 10.0];
        let ranges = vec![0.9, 0.9, 0.4, 0.5, 0.4, 0.5];

        let result = estimator.estimate(&measurements, &ranges);
        assert!(result.is_some());

        let (estimate, inliers) = result.unwrap();

        // Should still get 0.8383
        assert!((estimate - 0.8383).abs() < 0.001);

        // Last one should be outlier
        assert_eq!(inliers.len(), 6);
        assert!(inliers[0..5].iter().all(|&x| x));
        assert!(!inliers[5]);
    }

    #[test]
    fn test_tls_three_outliers() {
        let estimator = ScalarTLSEstimator::new();

        let measurements = vec![0.5, 1.0, 0.6, 20.0, 16.0, 10.0];
        let ranges = vec![0.9, 0.9, 0.4, 0.5, 0.4, 0.5];

        let result = estimator.estimate(&measurements, &ranges);
        assert!(result.is_some());

        let (estimate, inliers) = result.unwrap();

        // Reference: 0.6425
        assert!((estimate - 0.6425).abs() < 0.001);

        // First 3 inliers, last 3 outliers
        assert_eq!(inliers.len(), 6);
        assert!(inliers[0..3].iter().all(|&x| x));
        assert!(inliers[3..6].iter().all(|&x| !x));
    }
}
