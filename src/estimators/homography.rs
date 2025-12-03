//! Homography estimator using 4-point DLT-style algorithm.

use crate::core::Estimator;
use crate::models::Homography;
use crate::types::DataMatrix;
use crate::utils::gauss_elimination;
use nalgebra::{DMatrix, DVector, Matrix3};

/// Minimal homography estimator using a 4-point DLT-style algorithm.
pub struct HomographyEstimator;

impl Default for HomographyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl HomographyEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Estimate minimal model using Gaussian elimination (matches C++ implementation).
    fn estimate_minimal_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Homography> {
        // Build 8x9 augmented matrix [A | b] where we fix h[8] = 1.0
        // This matches the C++ implementation exactly
        let mut augmented = DMatrix::<f64>::zeros(8, 9);

        for (i, &idx) in sample.iter().enumerate() {
            let x1 = data[(idx, 0)];
            let y1 = data[(idx, 1)];
            let x2 = data[(idx, 2)];
            let y2 = data[(idx, 3)];

            // Row 2*i
            augmented[(2 * i, 0)] = -x1;
            augmented[(2 * i, 1)] = -y1;
            augmented[(2 * i, 2)] = -1.0;
            augmented[(2 * i, 3)] = 0.0;
            augmented[(2 * i, 4)] = 0.0;
            augmented[(2 * i, 5)] = 0.0;
            augmented[(2 * i, 6)] = x2 * x1;
            augmented[(2 * i, 7)] = x2 * y1;
            augmented[(2 * i, 8)] = -x2; // Inhomogeneous part

            // Row 2*i + 1
            augmented[(2 * i + 1, 0)] = 0.0;
            augmented[(2 * i + 1, 1)] = 0.0;
            augmented[(2 * i + 1, 2)] = 0.0;
            augmented[(2 * i + 1, 3)] = -x1;
            augmented[(2 * i + 1, 4)] = -y1;
            augmented[(2 * i + 1, 5)] = -1.0;
            augmented[(2 * i + 1, 6)] = y2 * x1;
            augmented[(2 * i + 1, 7)] = y2 * y1;
            augmented[(2 * i + 1, 8)] = -y2; // Inhomogeneous part
        }

        // Solve using Gaussian elimination
        let mut h = DVector::<f64>::zeros(8);
        if !gauss_elimination(&mut augmented, &mut h) {
            return Vec::new();
        }

        // Check for NaN
        if h.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Vec::new();
        }

        // Reshape into 3x3 matrix (h[8] = 1.0)
        let mut h_mat = Matrix3::<f64>::zeros();
        h_mat[(0, 0)] = h[0];
        h_mat[(0, 1)] = h[1];
        h_mat[(0, 2)] = h[2];
        h_mat[(1, 0)] = h[3];
        h_mat[(1, 1)] = h[4];
        h_mat[(1, 2)] = h[5];
        h_mat[(2, 0)] = h[6];
        h_mat[(2, 1)] = h[7];
        h_mat[(2, 2)] = 1.0;

        vec![Homography::new(h_mat)]
    }
}

impl Estimator for HomographyEstimator {
    type Model = Homography;

    fn sample_size(&self) -> usize {
        4
    }

    fn is_valid_sample(&self, _data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < self.sample_size() {
            return false;
        }
        // Ensure all indices are distinct.
        for i in 0..sample.len() {
            for j in (i + 1)..sample.len() {
                if sample[i] == sample[j] {
                    return false;
                }
            }
        }
        true
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() {
            return Vec::new();
        }

        // For minimal case (4 points), use Gaussian elimination like C++
        if n == self.sample_size() {
            return self.estimate_minimal_model(data, sample);
        }

        // For non-minimal case, use the non-minimal solver
        self.estimate_model_nonminimal(data, sample, None)
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

        // Build the 2N x 8 coefficient matrix and 2N x 1 inhomogeneous vector
        // (fixing h[8] = 1.0 like C++)
        let mut coefficients = DMatrix::<f64>::zeros(2 * n, 8);
        let mut inhomogeneous = DVector::<f64>::zeros(2 * n);

        for (i, &idx) in sample.iter().enumerate() {
            let x1 = data[(idx, 0)];
            let y1 = data[(idx, 1)];
            let x2 = data[(idx, 2)];
            let y2 = data[(idx, 3)];

            let weight = weights.map(|w| w[idx]).unwrap_or(1.0);
            let minus_weight_x1 = -weight * x1;
            let minus_weight_y1 = -weight * y1;
            let weight_x2 = weight * x2;
            let weight_y2 = weight * y2;

            // Row 2*i
            coefficients[(2 * i, 0)] = minus_weight_x1;
            coefficients[(2 * i, 1)] = minus_weight_y1;
            coefficients[(2 * i, 2)] = -weight;
            coefficients[(2 * i, 3)] = 0.0;
            coefficients[(2 * i, 4)] = 0.0;
            coefficients[(2 * i, 5)] = 0.0;
            coefficients[(2 * i, 6)] = weight_x2 * x1;
            coefficients[(2 * i, 7)] = weight_x2 * y1;
            inhomogeneous[2 * i] = -weight_x2;

            // Row 2*i + 1
            coefficients[(2 * i + 1, 0)] = 0.0;
            coefficients[(2 * i + 1, 1)] = 0.0;
            coefficients[(2 * i + 1, 2)] = 0.0;
            coefficients[(2 * i + 1, 3)] = minus_weight_x1;
            coefficients[(2 * i + 1, 4)] = minus_weight_y1;
            coefficients[(2 * i + 1, 5)] = -weight;
            coefficients[(2 * i + 1, 6)] = weight_y2 * x1;
            coefficients[(2 * i + 1, 7)] = weight_y2 * y1;
            inhomogeneous[2 * i + 1] = -weight_y2;
        }

        // Solve using QR decomposition (like C++ colPivHouseholderQr)
        let qr = coefficients.col_piv_qr();
        let h = match qr.solve(&inhomogeneous) {
            Some(h) => h,
            None => return Vec::new(),
        };

        // Check for NaN
        if h.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Vec::new();
        }

        // Reshape into 3x3 matrix (h[8] = 1.0)
        let mut h_mat = Matrix3::<f64>::zeros();
        h_mat[(0, 0)] = h[0];
        h_mat[(0, 1)] = h[1];
        h_mat[(0, 2)] = h[2];
        h_mat[(1, 0)] = h[3];
        h_mat[(1, 1)] = h[4];
        h_mat[(1, 2)] = h[5];
        h_mat[(2, 0)] = h[6];
        h_mat[(2, 1)] = h[7];
        h_mat[(2, 2)] = 1.0;

        vec![Homography::new(h_mat)]
    }

    fn is_valid_model(
        &self,
        model: &Homography,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        let det = model.h.determinant().abs();
        let min_det = 1e-4;
        let max_det = 1e4;
        det > min_det && det < max_det
    }
}
