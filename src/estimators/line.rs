//! Line estimator for 2D line fitting.

use crate::core::Estimator;
use crate::models::Line;
use crate::types::DataMatrix;

/// Line estimator for 2D line fitting.
///
/// Estimates lines in the form ax + by + c = 0 from 2D points.
/// The line parameters are normalized so that a² + b² = 1.
pub struct LineEstimator;

impl Default for LineEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl LineEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Estimator for LineEstimator {
    type Model = Line;

    fn sample_size(&self) -> usize {
        2 // A line requires 2 points
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < self.sample_size() {
            return false;
        }
        // Check for distinct indices
        for i in 0..sample.len() {
            for j in (i + 1)..sample.len() {
                if sample[i] == sample[j] {
                    return false;
                }
            }
        }
        // Data should have at least 2 columns (x, y)
        if data.ncols() < 2 {
            return false;
        }
        // Check that points are not too close together (avoid degenerate case)
        if sample.len() >= 2 {
            let idx1 = sample[0];
            let idx2 = sample[1];
            if idx1 >= data.nrows() || idx2 >= data.nrows() {
                return false;
            }
            let dx = data[(idx1, 0)] - data[(idx2, 0)];
            let dy = data[(idx1, 1)] - data[(idx2, 1)];
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < 1e-10 {
                return false; // Points are too close
            }
        }
        true
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        if sample.len() < self.sample_size() {
            return Vec::new();
        }

        // For minimal case (2 points), compute line directly
        if sample.len() == self.sample_size() {
            let idx1 = sample[0];
            let idx2 = sample[1];
            if idx1 >= data.nrows() || idx2 >= data.nrows() || data.ncols() < 2 {
                return Vec::new();
            }

            let x1 = data[(idx1, 0)];
            let y1 = data[(idx1, 1)];
            let x2 = data[(idx2, 0)];
            let y2 = data[(idx2, 1)];

            // Compute line parameters using cross product of homogeneous points
            // Line through (x1, y1) and (x2, y2) is given by cross product:
            // [x1, y1, 1] × [x2, y2, 1] = [a, b, c]
            let a = y1 - y2;
            let b = x2 - x1;
            let c = x1 * y2 - x2 * y1;

            // Normalize
            let norm = (a * a + b * b).sqrt();
            if norm < 1e-10 {
                return Vec::new(); // Degenerate line (points are identical)
            }

            vec![Line::new(a, b, c)]
        } else {
            // For non-minimal case, use least squares
            self.estimate_model_nonminimal(data, sample, None)
        }
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() {
            return Vec::new();
        }

        // Least squares fitting: minimize sum of squared distances
        // Distance from point (x, y) to line ax + by + c = 0 is |ax + by + c|
        // We want to minimize sum((ax_i + by_i + c)^2) subject to a² + b² = 1
        // This is equivalent to finding the eigenvector of the covariance matrix

        // Compute weighted centroid
        let mut sum_w = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;

        for &idx in sample {
            if idx >= data.nrows() || data.ncols() < 2 {
                continue;
            }
            let w = weights.map(|w| w[idx]).unwrap_or(1.0);
            sum_w += w;
            cx += w * data[(idx, 0)];
            cy += w * data[(idx, 1)];
        }

        if sum_w < 1e-10 {
            return Vec::new();
        }

        cx /= sum_w;
        cy /= sum_w;

        // Compute covariance matrix
        let mut cov_xx = 0.0;
        let mut cov_xy = 0.0;
        let mut cov_yy = 0.0;

        for &idx in sample {
            if idx >= data.nrows() || data.ncols() < 2 {
                continue;
            }
            let w = weights.map(|w| w[idx]).unwrap_or(1.0);
            let dx = data[(idx, 0)] - cx;
            let dy = data[(idx, 1)] - cy;
            cov_xx += w * dx * dx;
            cov_xy += w * dx * dy;
            cov_yy += w * dy * dy;
        }

        // The line normal is the eigenvector corresponding to the smallest eigenvalue
        // of the covariance matrix [cov_xx, cov_xy; cov_xy, cov_yy]
        // For a 2x2 matrix, we can solve directly
        let trace = cov_xx + cov_yy;
        let det = cov_xx * cov_yy - cov_xy * cov_xy;

        // Eigenvalues: (trace ± sqrt(trace² - 4*det)) / 2
        let discriminant = trace * trace - 4.0 * det;
        if discriminant < 0.0 {
            return Vec::new();
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;

        // Use the eigenvector corresponding to the smaller eigenvalue
        let (a, b) = if lambda2 < lambda1 {
            // Eigenvector for lambda2: solve (cov_xx - lambda2) * a + cov_xy * b = 0
            if cov_xy.abs() > 1e-10 {
                let b_val = 1.0;
                let a_val = -cov_xy / (cov_xx - lambda2).max(1e-10);
                (a_val, b_val)
            } else {
                // If cov_xy is zero, eigenvectors are [1, 0] and [0, 1]
                if cov_xx < cov_yy {
                    (1.0, 0.0)
                } else {
                    (0.0, 1.0)
                }
            }
        } else {
            // Eigenvector for lambda1
            if cov_xy.abs() > 1e-10 {
                let b_val = 1.0;
                let a_val = -cov_xy / (cov_xx - lambda1).max(1e-10);
                (a_val, b_val)
            } else if cov_xx < cov_yy {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        };

        // Normalize
        let norm = (a * a + b * b).sqrt();
        if norm < 1e-10 {
            return Vec::new();
        }
        let a_norm = a / norm;
        let b_norm = b / norm;

        // Compute c such that the line passes through the centroid
        // a*cx + b*cy + c = 0 => c = -(a*cx + b*cy)
        let c = -(a_norm * cx + b_norm * cy);

        vec![Line::new(a_norm, b_norm, c)]
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        // Check that the line parameters are normalized
        let norm_sq = model.params[0] * model.params[0] + model.params[1] * model.params[1];
        (norm_sq - 1.0).abs() < 1e-6
    }
}
