//! Essential matrix estimator using calibrated five-point hypotheses.

use crate::core::Estimator;
use crate::estimators::fundamental::FundamentalEstimator;
use crate::models::EssentialMatrix;
use crate::nister_stewenius::five_points_relative_pose;
use crate::types::DataMatrix;
use nalgebra::{Matrix3, SVD, UnitVector3, Vector3};

/// Essential matrix estimator using Nister-Stewenius five-point hypotheses and
/// constrained eight-point non-minimal refinement.
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
        5
    }

    fn non_minimal_sample_size(&self) -> usize {
        8
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < self.sample_size() || data.n_dims() < 4 {
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

        sample.iter().all(|&index| {
            index < data.n_points() && (0..4).all(|column| data.get(index, column).is_finite())
        })
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() {
            return Vec::new();
        }

        if n == self.sample_size() {
            let points1: [UnitVector3<f64>; 5] = std::array::from_fn(|index| {
                let row = sample[index];
                UnitVector3::new_normalize(Vector3::new(data.get(row, 0), data.get(row, 1), 1.0))
            });
            let points2: [UnitVector3<f64>; 5] = std::array::from_fn(|index| {
                let row = sample[index];
                UnitVector3::new_normalize(Vector3::new(data.get(row, 2), data.get(row, 3), 1.0))
            });
            return five_points_relative_pose(&points1, &points2)
                .filter(|model| model.e.iter().all(|value| value.is_finite()))
                .collect();
        }

        // Local optimization and final fitting use a constrained eight-point
        // estimate, which is more stable than reusing a minimal five-point fit.
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
        if sample.len() < self.non_minimal_sample_size() {
            return Vec::new();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_validation_only_inspects_the_minimal_sample() {
        let mut data = DataMatrix::zeros(6, 4);
        for row in 0..5 {
            data.set(row, 0, row as f64 * 0.1);
            data.set(row, 1, row as f64 * -0.2);
            data.set(row, 2, row as f64 * 0.3);
            data.set(row, 3, row as f64 * -0.4);
        }
        data.set(5, 0, f64::NAN);

        let estimator = EssentialEstimator::new();
        assert!(estimator.is_valid_sample(&data, &[0, 1, 2, 3, 4]));
        assert!(!estimator.is_valid_sample(&data, &[0, 1, 2, 3, 5]));
    }
}
