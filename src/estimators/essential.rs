//! Essential matrix estimator using a constrained eight-point solver.

use crate::core::Estimator;
use crate::estimators::fundamental::FundamentalEstimator;
use crate::models::EssentialMatrix;
use crate::types::DataMatrix;
use nalgebra::{Matrix3, SVD};

/// Essential matrix estimator using the eight-point algorithm with essential constraints.
///
/// The Nister-Stewenius five-point action-matrix solver is retained separately
/// for future work, but the public estimator uses the stable constrained
/// eight-point path.
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
        8
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

        (0..data.n_points()).all(|index| (0..4).all(|column| data.get(index, column).is_finite()))
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() {
            return Vec::new();
        }

        // Use the constrained eight-point solver for stable minimal estimates.
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
