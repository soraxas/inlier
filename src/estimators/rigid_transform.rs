//! Rigid transform estimator using Procrustes analysis.

use crate::core::Estimator;
use crate::models::RigidTransform;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Matrix3, SVD};

/// Rigid transform estimator using Procrustes analysis.
#[derive(Clone)]
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

    fn point(data: &DataMatrix, index: usize, offset: usize) -> Option<nalgebra::Vector3<f64>> {
        if index >= data.n_points() || data.n_dims() < offset + 3 {
            return None;
        }
        let point = nalgebra::Vector3::new(
            data.get(index, offset),
            data.get(index, offset + 1),
            data.get(index, offset + 2),
        );
        point.iter().all(|value| value.is_finite()).then_some(point)
    }

    fn forms_triangle(
        first: nalgebra::Vector3<f64>,
        second: nalgebra::Vector3<f64>,
        third: nalgebra::Vector3<f64>,
    ) -> bool {
        let first_edge = second - first;
        let second_edge = third - first;
        let scale = first_edge.norm().max(second_edge.norm());
        scale.is_finite()
            && scale > 1e-12
            && first_edge.cross(&second_edge).norm() > 1e-10 * scale * scale
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
        // A rigid transform is not uniquely determined by collinear points.
        if data.n_dims() < 6 {
            return false;
        }
        if sample.iter().any(|&index| {
            Self::point(data, index, 0).is_none() || Self::point(data, index, 3).is_none()
        }) {
            return false;
        }
        let Some(source0) = Self::point(data, sample[0], 0) else {
            return false;
        };
        let Some(source1) = Self::point(data, sample[1], 0) else {
            return false;
        };
        let Some(source2) = Self::point(data, sample[2], 0) else {
            return false;
        };
        let Some(target0) = Self::point(data, sample[0], 3) else {
            return false;
        };
        let Some(target1) = Self::point(data, sample[1], 3) else {
            return false;
        };
        let Some(target2) = Self::point(data, sample[2], 3) else {
            return false;
        };

        Self::forms_triangle(source0, source1, source2)
            && Self::forms_triangle(target0, target1, target2)
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() || !self.is_valid_sample(data, sample) {
            return Vec::new();
        }

        use nalgebra::Vector3;

        // Compute centroids
        let mut c0 = Vector3::<f64>::zeros();
        let mut c1 = Vector3::<f64>::zeros();

        for &idx in sample {
            c0[0] += data.get(idx, 0);
            c0[1] += data.get(idx, 1);
            c0[2] += data.get(idx, 2);
            c1[0] += data.get(idx, 3);
            c1[1] += data.get(idx, 4);
            c1[2] += data.get(idx, 5);
        }

        c0 /= n as f64;
        c1 /= n as f64;

        // Build centered point matrices
        let mut p0 = DMatrix::<f64>::zeros(3, n);
        let mut p1 = DMatrix::<f64>::zeros(3, n);

        let mut avg_dist0 = 0.0;
        let mut avg_dist1 = 0.0;

        for (col, &idx) in sample.iter().enumerate() {
            p0[(0, col)] = data.get(idx, 0) - c0[0];
            p0[(1, col)] = data.get(idx, 1) - c0[1];
            p0[(2, col)] = data.get(idx, 2) - c0[2];

            p1[(0, col)] = data.get(idx, 3) - c1[0];
            p1[(1, col)] = data.get(idx, 4) - c1[1];
            p1[(2, col)] = data.get(idx, 5) - c1[2];

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
        let (Some(u), Some(vt)) = (svd.u, svd.v_t) else {
            return Vec::new();
        };
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
        model.rotation.coords.iter().all(|value| value.is_finite())
            && model
                .translation
                .vector
                .iter()
                .all(|value| value.is_finite())
            && det.is_finite()
            && (det - 1.0).abs() < 1e-2 * _threshold.max(1.0)
    }
}
