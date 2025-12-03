//! Essential matrix estimator using 5-point Nister-Stewenius algorithm.

use crate::core::Estimator;
use crate::estimators::fundamental::FundamentalEstimator;
use crate::models::{EssentialMatrix, FundamentalMatrix};
use crate::nister_stewenius::five_points_relative_pose;
use crate::types::DataMatrix;
use nalgebra::{Matrix3, SVD, UnitVector3, Vector3};

/// Essential matrix estimator using 8-point algorithm with essential matrix constraints.
/// Uses the 5-point Nister-Stewenius algorithm for minimal samples.
pub struct EssentialEstimator;

impl Default for EssentialEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl EssentialEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Five-point Nister-Stewenius solver for essential matrix.
    ///
    /// Uses the Nister-Stewenius implementation to solve for essential matrix
    /// from 5 point correspondences. Returns up to 10 solutions.
    fn estimate_five_point_nister(
        &self,
        data: &DataMatrix,
        sample: &[usize],
    ) -> Vec<EssentialMatrix> {
        if sample.len() != 5 {
            return Vec::new();
        }

        // Extract 5 point correspondences and convert to normalized 3D vectors
        // Data format: [x1, y1, x2, y2] for each row
        // For essential matrix, we assume calibrated cameras (normalized coordinates)
        let mut a = [UnitVector3::new_unchecked(Vector3::new(0.0, 1.0, 0.0)); 5];
        let mut b = [UnitVector3::new_unchecked(Vector3::new(0.0, 1.0, 0.0)); 5];

        for (i, &idx) in sample.iter().enumerate() {
            if idx >= data.nrows() || data.ncols() < 4 {
                return Vec::new();
            }

            // Extract 2D points
            let x1 = data[(idx, 0)];
            let y1 = data[(idx, 1)];
            let x2 = data[(idx, 2)];
            let y2 = data[(idx, 3)];

            // Convert to normalized 3D homogeneous coordinates (calibrated camera)
            // For essential matrix, we assume normalized image coordinates
            // If the data is not normalized, we should normalize it first
            let v1 = Vector3::new(x1, y1, 1.0);
            let v2 = Vector3::new(x2, y2, 1.0);

            // Normalize to unit vectors
            let norm1 = v1.norm();
            if norm1 < 1e-10 {
                return Vec::new(); // Degenerate point
            }
            a[i] = UnitVector3::new_unchecked(v1 / norm1);

            let norm2 = v2.norm();
            if norm2 < 1e-10 {
                return Vec::new(); // Degenerate point
            }
            b[i] = UnitVector3::new_unchecked(v2 / norm2);
        }

        // Call the Nister-Stewenius solver
        five_points_relative_pose(&a, &b).collect()
    }

    /// Enforce essential matrix constraints: det(E) = 0 and two equal singular values.
    fn enforce_essential_constraints(&self, f: Matrix3<f64>) -> Matrix3<f64> {
        // SVD: E = U * diag(s1, s2, s3) * V^T
        // For essential matrix: s1 = s2, s3 = 0
        let svd = SVD::new(f, true, true);
        let u = svd.u.unwrap();
        let s = svd.singular_values;
        let vt = svd.v_t.unwrap();

        // Average the first two singular values and set third to zero
        let avg_s = (s[0] + s[1]) / 2.0;
        let mut s_essential = nalgebra::Vector3::<f64>::zeros();
        s_essential[0] = avg_s;
        s_essential[1] = avg_s;
        s_essential[2] = 0.0;

        // Reconstruct: E = U * diag(avg_s, avg_s, 0) * V^T
        let s_diag = nalgebra::Matrix3::<f64>::from_diagonal(&s_essential);
        u * s_diag * vt
    }
}

impl Estimator for EssentialEstimator {
    type Model = EssentialMatrix;

    fn sample_size(&self) -> usize {
        5 // Now using 5-point Nister solver
    }

    fn is_valid_sample(&self, _data: &DataMatrix, sample: &[usize]) -> bool {
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
        true
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() {
            return Vec::new();
        }

        // Use 5-point Nister solver for exactly 5 points
        if n == 5 {
            return self.estimate_five_point_nister(data, sample);
        }

        // For 6+ points, use 8-point + constraints
        let fundamental_est = FundamentalEstimator::new();
        let f_models = fundamental_est.estimate_model(data, sample);

        if f_models.is_empty() {
            return Vec::new();
        }

        // Convert fundamental matrix to essential matrix by enforcing constraints
        let f = f_models[0].f;
        let e = self.enforce_essential_constraints(f);

        vec![EssentialMatrix::new(e)]
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        // Use fundamental matrix estimator and then enforce essential constraints
        let fundamental_est = FundamentalEstimator::new();
        let f_models = fundamental_est.estimate_model_nonminimal(data, sample, weights);

        if f_models.is_empty() {
            return Vec::new();
        }

        // Convert fundamental matrix to essential matrix by enforcing constraints
        let f = f_models[0].f;
        let e = self.enforce_essential_constraints(f);

        vec![EssentialMatrix::new(e)]
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        // Essential matrix should have determinant = 0 and two equal singular values
        let det = model.e.determinant().abs();
        det < 1e-3 * _threshold.max(1.0)
    }
}
