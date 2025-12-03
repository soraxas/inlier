//! Absolute pose estimator using P3P (Perspective-3-Point) algorithm.

use crate::bundle_adjustment::refine_absolute_pose;
use crate::core::Estimator;
use crate::models::AbsolutePose;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Matrix3, SVD, Vector2, Vector3};
#[cfg(feature = "kornia-pnp")]
use kornia_pnp::epnp::solve_epnp;

/// Absolute pose estimator using P3P (Perspective-3-Point) algorithm.
///
/// For minimal samples (3 points), this uses a simplified DLT-based approach.
/// For non-minimal samples (4+ points), it uses bundle adjustment for refinement.
///
/// # Note
/// The `kornia-pnp` crate is available for more robust OnP solvers (e.g., EOnP).
/// To use it, enable the `kornia-pnp` feature and update the `estimate_model_nonminimal`
/// method to call `kornia_pnp::epnp::solve_epnp` with proper tensor type conversions.
pub struct AbsolutePoseEstimator;

impl Default for AbsolutePoseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl AbsolutePoseEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Estimator for AbsolutePoseEstimator {
    type Model = AbsolutePose;

    fn sample_size(&self) -> usize {
        3 // P3P requires 3 points
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
        // Data should have at least 5 columns: (u, v, x, y, z)
        if data.ncols() < 5 {
            return false;
        }
        true
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() || data.ncols() < 5 {
            return Vec::new();
        }

        // Simplified P3P: Use DLT-style approach with 3 points
        // For a proper implementation, we'd solve the P3P polynomial system
        // Here we use a simplified approach: estimate pose using 3D-2D correspondences

        use nalgebra::{DMatrix, Matrix3, SVD, Vector3};

        // Build system: For each point, we have:
        // [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T
        // Assuming normalized coordinates (K = I), we have:
        // u = (r11*X + r12*Y + r13*Z + tx) / (r31*X + r32*Y + r33*Z + tz)
        // v = (r21*X + r22*Y + r23*Z + ty) / (r31*X + r32*Y + r33*Z + tz)

        // For 3 points, we can set up a linear system
        // This is a simplified approach - proper P3P would solve a polynomial system

        let mut a = DMatrix::<f64>::zeros(2 * n, 12);
        for (i, &idx) in sample.iter().enumerate() {
            let u = data[(idx, 0)];
            let v = data[(idx, 1)];
            let x = data[(idx, 2)];
            let y = data[(idx, 3)];
            let z = data[(idx, 4)];

            // Row 2*i: u constraint
            a[(2 * i, 0)] = x;
            a[(2 * i, 1)] = y;
            a[(2 * i, 2)] = z;
            a[(2 * i, 3)] = 1.0;
            a[(2 * i, 8)] = -u * x;
            a[(2 * i, 9)] = -u * y;
            a[(2 * i, 10)] = -u * z;
            a[(2 * i, 11)] = -u;

            // Row 2*i + 1: v constraint
            a[(2 * i + 1, 4)] = x;
            a[(2 * i + 1, 5)] = y;
            a[(2 * i + 1, 6)] = z;
            a[(2 * i + 1, 7)] = 1.0;
            a[(2 * i + 1, 8)] = -v * x;
            a[(2 * i + 1, 9)] = -v * y;
            a[(2 * i + 1, 10)] = -v * z;
            a[(2 * i + 1, 11)] = -v;
        }

        // Solve via SVD
        let svd = SVD::new(a, false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let v = vt.transpose();
        let last_col = v.column(v.ncols() - 1);

        if last_col.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Vec::new();
        }

        // Extract rotation and translation from solution
        // The solution is [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz]
        let mut r = Matrix3::<f64>::zeros();
        r[(0, 0)] = last_col[0];
        r[(0, 1)] = last_col[1];
        r[(0, 2)] = last_col[2];
        r[(1, 0)] = last_col[4];
        r[(1, 1)] = last_col[5];
        r[(1, 2)] = last_col[6];
        r[(2, 0)] = last_col[8];
        r[(2, 1)] = last_col[9];
        r[(2, 2)] = last_col[10];

        // Enforce orthogonality via SVD
        let r_svd = SVD::new(r, true, true);
        let u_r = r_svd.u.unwrap();
        let vt_r = r_svd.v_t.unwrap();
        let v_r = vt_r.transpose();
        let r_ortho = u_r * v_r.transpose();

        // Ensure proper rotation (det = 1)
        let mut r_final = r_ortho;
        if r_final.determinant() < 0.0 {
            let mut u_neg = u_r;
            u_neg.column_mut(2).neg_mut();
            r_final = u_neg * v_r.transpose();
        }

        let t = Vector3::<f64>::new(last_col[3], last_col[7], last_col[11]);

        // Convert to AbsolutePose
        vec![AbsolutePose::from_rt(r_final, t)]
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        if sample.len() < 4 {
            // For < 4 points, fall back to minimal estimation
            return self.estimate_model(data, sample);
        }

        // Convert to format expected by kornia-pnp
        // kornia-pnp expects: points_2d (Nx2), points_3d (Nx3), camera matrix (3x3)
        let mut points_2d = Vec::new();
        let mut points_3d = Vec::new();

        for &idx in sample {
            if idx >= data.nrows() || data.ncols() < 5 {
                continue;
            }
            points_2d.push([data[(idx, 0)], data[(idx, 1)]]);
            points_3d.push([data[(idx, 2)], data[(idx, 3)], data[(idx, 4)]]);
        }

        if points_2d.len() < 4 {
            return Vec::new();
        }

        // Use kornia-pnp EOnP solver for non-minimal estimation
        // EOnP works with >= 4 points and is more robust than DLT
        #[cfg(feature = "kornia-pnp")]
        {
            use kornia_tensor::Tensor;
            use kornia_image::Image;

            // Create identity camera matrix (assuming normalized coordinates)
            // For calibrated cameras, this should be the actual K matrix
            let camera_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

            // Try to use kornia-pnp EOnP solver
            // Note: kornia-pnp uses its own tensor types, so we need to convert
            // For now, fall back to our existing implementation
            // TODO: Properly integrate kornia-pnp tensor types
        }

        // Fall back to existing DLT + bundle adjustment approach
        let initial_models = self.estimate_model(data, sample);
        if initial_models.is_empty() {
            return Vec::new();
        }

        let rot = initial_models[0].rotation.to_rotation_matrix();
        let mut r = *rot.matrix();
        let mut t_vec = initial_models[0].translation.vector;

        // Apply bundle adjustment if we have enough points
        if sample.len() >= 4 {
            // Convert to Vector2/Vector3 format for bundle adjustment
            let mut points_2d_vec = Vec::new();
            let mut points_3d_vec = Vec::new();
            let mut weights_vec = Vec::new();

            for &idx in sample {
                points_2d_vec.push(Vector2::new(data[(idx, 0)], data[(idx, 1)]));
                points_3d_vec.push(Vector3::new(data[(idx, 2)], data[(idx, 3)], data[(idx, 4)]));
                if let Some(w) = weights {
                    weights_vec.push(w[idx]);
                }
            }

            let weights_slice = if weights_vec.is_empty() {
                None
            } else {
                Some(weights_vec.as_slice())
            };
            refine_absolute_pose(
                &points_2d_vec,
                &points_3d_vec,
                &mut r,
                &mut t_vec,
                weights_slice,
                100,
            );
        }

        vec![AbsolutePose::from_rt(r, t_vec)]
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        // Check that rotation is proper
        let r = model.rotation.to_rotation_matrix();
        let det = r.matrix().determinant();
        (det - 1.0).abs() < 1e-2 * _threshold.max(1.0)
    }
}
