//! Similarity (scale + rotation + translation) estimator using Umeyama/Procrustes.

use crate::core::Estimator;
use crate::models::SimilarityTransform;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Matrix3, SVD};

#[derive(Clone)]
pub struct SimilarityTransformEstimator;

impl Default for SimilarityTransformEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityTransformEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Estimator for SimilarityTransformEstimator {
    type Model = SimilarityTransform;

    fn sample_size(&self) -> usize {
        3
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < self.sample_size() || data.n_dims() < 6 {
            return false;
        }
        // Distinct indices
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
        if n < self.sample_size() || data.n_dims() < 6 {
            return Vec::new();
        }

        // Gather centroids.
        let mut c0 = nalgebra::Vector3::<f64>::zeros();
        let mut c1 = nalgebra::Vector3::<f64>::zeros();
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

        // Build centered matrices.
        let mut p0 = DMatrix::<f64>::zeros(3, n);
        let mut p1 = DMatrix::<f64>::zeros(3, n);

        for (col, &idx) in sample.iter().enumerate() {
            p0[(0, col)] = data.get(idx, 0) - c0[0];
            p0[(1, col)] = data.get(idx, 1) - c0[1];
            p0[(2, col)] = data.get(idx, 2) - c0[2];

            p1[(0, col)] = data.get(idx, 3) - c1[0];
            p1[(1, col)] = data.get(idx, 4) - c1[1];
            p1[(2, col)] = data.get(idx, 5) - c1[2];
        }

        let var0 = p0.column_iter().map(|c| c.norm_squared()).sum::<f64>() / n as f64;
        if !var0.is_finite() || var0 <= 0.0 {
            return Vec::new();
        }

        // Cross-covariance.
        let sigma = &p1 * &p0.transpose() / (n as f64);
        let sigma_fixed = Matrix3::<f64>::from_iterator(sigma.iter().cloned());

        let svd = SVD::new(sigma, true, true);
        let (u_opt, v_t_opt) = match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => (u, vt),
            _ => return Vec::new(),
        };

        let mut r = &u_opt * &v_t_opt;
        if r.determinant() < 0.0 {
            let mut u_fix = u_opt.clone();
            u_fix.column_mut(2).neg_mut();
            r = &u_fix * &v_t_opt;
        }

        let r_fixed = Matrix3::<f64>::from_iterator(r.iter().cloned());

        let s_num = (r_fixed.transpose() * sigma_fixed).trace();
        let scale = (s_num / var0).max(0.0);

        // translation: t = c1 - s * R * c0
        let t = c1 - scale * (r_fixed * c0);

        let rot3 = nalgebra::Rotation3::from_matrix_unchecked(r_fixed);
        let quat = nalgebra::UnitQuaternion::from_rotation_matrix(&rot3);
        let trans = nalgebra::Translation3::from(t);

        vec![SimilarityTransform::new(scale, quat, trans)]
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        _weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        self.estimate_model(data, sample)
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        model.scale.is_finite()
            && model.scale > 0.0
            && (model.rotation.to_rotation_matrix().matrix().determinant() - 1.0).abs() < 1e-2
    }
}
