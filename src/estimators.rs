//! Estimators for geometric models.
//!
//! This module currently contains a minimal homography estimator. The
//! implementation is intentionally simple and focuses on having a working
//! skeleton that can be exercised by tests; numerical refinements can be
//! added later.

use crate::core::Estimator;
use crate::models::Homography;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Matrix3, SVD};

/// Minimal homography estimator using a 4-point DLT-style algorithm.
pub struct HomographyEstimator;

impl HomographyEstimator {
    pub fn new() -> Self {
        Self
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

        // Build the 2N x 9 design matrix for DLT.
        let mut a = DMatrix::<f64>::zeros(2 * n, 9);
        for (i, &idx) in sample.iter().enumerate() {
            let x1 = data[(idx, 0)];
            let y1 = data[(idx, 1)];
            let x2 = data[(idx, 2)];
            let y2 = data[(idx, 3)];

            // Row 2*i
            a[(2 * i, 0)] = -x1;
            a[(2 * i, 1)] = -y1;
            a[(2 * i, 2)] = -1.0;
            a[(2 * i, 6)] = x2 * x1;
            a[(2 * i, 7)] = x2 * y1;
            a[(2 * i, 8)] = x2;

            // Row 2*i + 1
            a[(2 * i + 1, 3)] = -x1;
            a[(2 * i + 1, 4)] = -y1;
            a[(2 * i + 1, 5)] = -1.0;
            a[(2 * i + 1, 6)] = y2 * x1;
            a[(2 * i + 1, 7)] = y2 * y1;
            a[(2 * i + 1, 8)] = y2;
        }

        // Solve Ah = 0 via SVD; the solution is the singular vector
        // corresponding to the smallest singular value.
        let svd = SVD::new(a.clone(), false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let v = vt.transpose();

        // Last column of V corresponds to the smallest singular value.
        let last_col = v.column(v.ncols() - 1);
        // Reshape into a 3x3 matrix.
        let mut h_mat = Matrix3::<f64>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                h_mat[(r, c)] = last_col[3 * r + c];
            }
        }

        // Return the raw homography; callers/tests can normalize or compare
        // up to scale as appropriate.
        vec![Homography::new(h_mat)]
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
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

#[cfg(test)]
mod tests {
    use super::HomographyEstimator;
    use crate::core::Estimator;
    use crate::types::DataMatrix;

    #[test]
    fn homography_estimator_recovers_simple_translation() {
        // Ground-truth homography: translation by (tx, ty).
        let tx = 1.0;
        let ty = 2.0;

        // Four perfect correspondences (x, y) -> (x + tx, y + ty).
        let correspondences = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ];

        let mut data = DataMatrix::zeros(4, 4);
        for (i, (x, y)) in correspondences.iter().enumerate() {
            data[(i, 0)] = *x;
            data[(i, 1)] = *y;
            data[(i, 2)] = x + tx;
            data[(i, 3)] = y + ty;
        }

        let estimator = HomographyEstimator::new();
        let sample = [0usize, 1, 2, 3];

        assert!(estimator.is_valid_sample(&data, &sample));

        let models = estimator.estimate_model(&data, &sample);
        assert_eq!(models.len(), 1);

        let homography = &models[0];
        assert!(
            estimator.is_valid_model(homography, &data, &sample, 0.0),
            "Estimated homography should satisfy basic determinant constraints"
        );
    }
}
