//! 3-D plane estimator for point clouds.
//!
//! Fits planes of the form `ax + by + cz + d = 0` (unit normal).
//!
//! * **Minimal** (3 points): cross product of two edge vectors.
//! * **Non-minimal** (≥3 points, optional weights): PCA via 3×3 covariance SVD.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::core::Estimator;
use crate::models::Plane3;
use crate::types::DataMatrix;

pub struct PlaneEstimator;

impl Default for PlaneEstimator {
    fn default() -> Self {
        Self
    }
}

impl PlaneEstimator {
    pub fn new() -> Self {
        Self
    }

    fn point3(data: &DataMatrix, idx: usize) -> Vector3<f64> {
        Vector3::new(data.get(idx, 0), data.get(idx, 1), data.get(idx, 2))
    }

    fn finite_point(data: &DataMatrix, idx: usize) -> Option<Vector3<f64>> {
        let p = Self::point3(data, idx);
        p.iter().all(|v| v.is_finite()).then_some(p)
    }

    fn weight(weights: Option<&[f64]>, idx: usize) -> Option<f64> {
        match weights {
            Some(weights) => {
                let w = *weights.get(idx)?;
                (w.is_finite() && w >= 0.0).then_some(w)
            }
            None => Some(1.0),
        }
    }

    fn fit_pca(data: &DataMatrix, sample: &[usize], weights: Option<&[f64]>) -> Option<Plane3> {
        let mut sum_w = 0.0_f64;
        let mut centroid = Vector3::zeros();

        for &idx in sample {
            let w = Self::weight(weights, idx)?;
            let p = Self::finite_point(data, idx)?;
            centroid += w * p;
            sum_w += w;
        }
        if sum_w < 1e-10 {
            return None;
        }
        centroid /= sum_w;

        let mut cov = Matrix3::zeros();
        for &idx in sample {
            let w = Self::weight(weights, idx)?;
            let d = Self::finite_point(data, idx)? - centroid;
            cov += w * d * d.transpose();
        }
        if cov.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let eig = SymmetricEigen::new(cov);
        // Plane normal = eigenvector of the SMALLEST eigenvalue.
        let min_col = (0..3)
            .min_by(|&i, &j| eig.eigenvalues[i].partial_cmp(&eig.eigenvalues[j]).unwrap())
            .unwrap_or(0);
        let normal: Vector3<f64> = eig.eigenvectors.column(min_col).into();
        let norm = normal.norm();
        if norm < 1e-10 {
            return None;
        }
        Some(Plane3::from_normal_and_point(normal / norm, centroid))
    }
}

impl Estimator for PlaneEstimator {
    type Model = Plane3;

    fn sample_size(&self) -> usize {
        3
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < 3 || data.n_dims() < 3 {
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
        // Non-collinear: cross product magnitude above threshold
        let Some(p0) = Self::finite_point(data, sample[0]) else {
            return false;
        };
        let Some(p1) = Self::finite_point(data, sample[1]) else {
            return false;
        };
        let Some(p2) = Self::finite_point(data, sample[2]) else {
            return false;
        };
        (p1 - p0).cross(&(p2 - p0)).norm() > 1e-10
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Plane3> {
        if sample.len() < 3 || data.n_dims() < 3 {
            return vec![];
        }
        let Some(p0) = Self::finite_point(data, sample[0]) else {
            return vec![];
        };
        let Some(p1) = Self::finite_point(data, sample[1]) else {
            return vec![];
        };
        let Some(p2) = Self::finite_point(data, sample[2]) else {
            return vec![];
        };
        let cross = (p1 - p0).cross(&(p2 - p0));
        let norm = cross.norm();
        if !norm.is_finite() || norm < 1e-10 {
            return vec![];
        }
        vec![Plane3::from_normal_and_point(cross / norm, p0)]
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Plane3> {
        Self::fit_pca(data, sample, weights).into_iter().collect()
    }

    fn is_valid_model(
        &self,
        model: &Plane3,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        (model.normal.norm() - 1.0).abs() < 1e-5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plane_data(n: usize, normal: Vector3<f64>, d: f64, noise: f64) -> DataMatrix {
        use rand::{Rng, SeedableRng, rngs::SmallRng};
        let mut rng = SmallRng::seed_from_u64(42);
        let mut data = DataMatrix::zeros(n, 3);
        // Pick two vectors orthogonal to normal
        let u = if normal.x.abs() < 0.9 {
            normal.cross(&Vector3::x()).normalize()
        } else {
            normal.cross(&Vector3::y()).normalize()
        };
        let v = normal.cross(&u).normalize();
        for i in 0..n {
            let s: f64 = rng.random_range(-5.0..5.0);
            let t: f64 = rng.random_range(-5.0..5.0);
            // Point on the plane: p · normal + d = 0 → p = s*u + t*v - (d/1)*normal
            // Simpler: move along u/v and offset along normal by -d
            let base = s * u + t * v;
            let on_plane = base - normal * (normal.dot(&base) + d);
            data.set(i, 0, on_plane.x + rng.random_range(-noise..noise));
            data.set(i, 1, on_plane.y + rng.random_range(-noise..noise));
            data.set(i, 2, on_plane.z + rng.random_range(-noise..noise));
        }
        data
    }

    #[test]
    fn minimal_fit_xy_plane() {
        let mut data = DataMatrix::zeros(3, 3);
        data.set(0, 0, 0.0);
        data.set(0, 1, 0.0);
        data.set(0, 2, 0.0);
        data.set(1, 0, 1.0);
        data.set(1, 1, 0.0);
        data.set(1, 2, 0.0);
        data.set(2, 0, 0.0);
        data.set(2, 1, 1.0);
        data.set(2, 2, 0.0);

        let est = PlaneEstimator::new();
        let models = est.estimate_model(&data, &[0, 1, 2]);
        assert_eq!(models.len(), 1);
        let plane = &models[0];
        assert!(
            (plane.normal.z.abs() - 1.0).abs() < 1e-6,
            "normal should be Z: {:?}",
            plane.normal
        );
        assert!(plane.distance(0.5, 0.5, 0.0) < 1e-6);
    }

    #[test]
    fn pca_fit_tilted_plane() {
        let normal = Vector3::new(1.0_f64, 1.0, 1.0).normalize();
        let d = -2.0;
        let data = make_plane_data(200, normal, d, 0.001);
        let sample: Vec<usize> = (0..200).collect();
        let est = PlaneEstimator::new();
        let models = est.estimate_model_nonminimal(&data, &sample, None);
        assert_eq!(models.len(), 1);
        let plane = &models[0];
        let dot = plane.normal.dot(&normal).abs();
        assert!(
            dot > 0.999,
            "normal should align with ground truth: dot={dot}"
        );
    }
}
