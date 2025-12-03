//! Rigid transform estimator using Procrustes analysis.

use crate::core::Estimator;
use crate::models::RigidTransform;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Matrix3, SVD};

/// Rigid transform estimator using Procrustes analysis.
pub struct RigidTransformEstimator;

impl Default for RigidTransformEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl RigidTransformEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Estimator for RigidTransformEstimator {
    type Model = RigidTransform;

    fn sample_size(&self) -> usize {
        3
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
        // Check for collinearity (simplified: just ensure we have enough points)
        if sample.len() < 3 {
            return false;
        }
        // Basic check: ensure data has 6 columns (x1,y1,z1,x2,y2,z2)
        if data.ncols() < 6 {
            return false;
        }
        true
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() || data.ncols() < 6 {
            return Vec::new();
        }

        use nalgebra::Vector3;

        // Compute centroids
        let mut c0 = Vector3::<f64>::zeros();
        let mut c1 = Vector3::<f64>::zeros();

        for &idx in sample {
            c0[0] += data[(idx, 0)];
            c0[1] += data[(idx, 1)];
            c0[2] += data[(idx, 2)];
            c1[0] += data[(idx, 3)];
            c1[1] += data[(idx, 4)];
            c1[2] += data[(idx, 5)];
        }

        c0 /= n as f64;
        c1 /= n as f64;

        // Build centered point matrices
        let mut p0 = DMatrix::<f64>::zeros(3, n);
        let mut p1 = DMatrix::<f64>::zeros(3, n);

        let mut avg_dist0 = 0.0;
        let mut avg_dist1 = 0.0;

        for (col, &idx) in sample.iter().enumerate() {
            p0[(0, col)] = data[(idx, 0)] - c0[0];
            p0[(1, col)] = data[(idx, 1)] - c0[1];
            p0[(2, col)] = data[(idx, 2)] - c0[2];

            p1[(0, col)] = data[(idx, 3)] - c1[0];
            p1[(1, col)] = data[(idx, 4)] - c1[1];
            p1[(2, col)] = data[(idx, 5)] - c1[2];

            avg_dist0 += p0.column(col).norm();
            avg_dist1 += p1.column(col).norm();
        }

        avg_dist0 /= n as f64;
        avg_dist1 /= n as f64;

        if avg_dist0 < 1e-10 || avg_dist1 < 1e-10 {
            return Vec::new();
        }

        // Normalize for numerical stability
        let s0 = 3.0_f64.sqrt() / avg_dist0;
        let s1 = 3.0_f64.sqrt() / avg_dist1;
        p0 *= s0;
        p1 *= s1;

        // Compute covariance matrix H = P0 * P1^T
        let h = &p0 * &p1.transpose();

        if h.iter().any(|&x| x.is_nan()) {
            return Vec::new();
        }

        // SVD: H = U * S * V^T, then R = V * U^T
        let svd = SVD::new(h, true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let v = vt.transpose();

        let mut r = &v * &u.transpose();

        // Ensure proper rotation (det(R) = 1)
        if r.determinant() < 0.0 {
            let mut v_neg = v.clone();
            v_neg.column_mut(2).neg_mut();
            r = &v_neg * &u.transpose();
        }

        // Convert to fixed-size matrix for Rotation3
        let r_fixed = Matrix3::<f64>::from_iterator(r.iter().cloned());

        // Compute translation: t = c1 - R * c0
        let t = c1 - r_fixed * c0;

        // Convert to UnitQuaternion and Translation3
        use nalgebra::{Rotation3, UnitQuaternion};
        let rot3 = Rotation3::from_matrix_unchecked(r_fixed);
        let quat = UnitQuaternion::from_rotation_matrix(&rot3);
        let translation = nalgebra::Translation3::from(t);

        vec![RigidTransform::new(quat, translation)]
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        _weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        // For non-minimal case, Procrustes analysis naturally handles more points
        // The same algorithm works for any number of points >= 3
        self.estimate_model(data, sample)
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        // Check that rotation is proper (determinant close to 1)
        let r = model.rotation.to_rotation_matrix();
        let det = r.matrix().determinant();
        (det - 1.0).abs() < 1e-2 * _threshold.max(1.0)
    }
}
