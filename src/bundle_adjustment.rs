//! Bundle adjustment and numerical optimization for geometric models.
//!
//! This module provides Levenberg-Marquardt optimization for refining
//! geometric models using argmin.

use argmin::core::{CostFunction, Gradient};
use nalgebra::{DVector, Matrix3, Quaternion, UnitQuaternion, Vector2, Vector3};

/// Sampson error for fundamental matrix refinement.
/// Minimizes the Sampson distance: r = (x2^T * F * x1) / ||J_C||
/// where J_C is the Jacobian of the epipolar constraint.
pub fn sampson_error(f: &Matrix3<f64>, x1: &Vector2<f64>, x2: &Vector2<f64>) -> f64 {
    // Epipolar constraint: x2^T * F * x1 = 0
    let x1_home = Vector3::new(x1.x, x1.y, 1.0);
    let x2_home = Vector3::new(x2.x, x2.y, 1.0);
    let c = (x2_home.transpose() * f * x1_home)[0];

    // Jacobian of constraint w.r.t. image points
    // J_C = [F.block<3,2>(0,0)^T * x2; F.block<2,3>(0,0) * x1] (4D vector)
    // This matches C++: J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(),
    //                      F.block<2, 3>(0, 0) * x1[k].homogeneous();
    // F.block<3,2>(0,0) is first 2 columns, F.block<2,3>(0,0) is first 2 rows
    let f_cols_01 = f.columns(0, 2); // First 2 columns (3x2)
    let f_rows_01 = f.rows(0, 2); // First 2 rows (2x3)

    let f_t_x2_part = f_cols_01.transpose() * x2_home; // 2x1
    let f_x1_part = f_rows_01 * x1_home; // 2x1

    // Build 4D J_C vector: [f_t_x2_part[0], f_t_x2_part[1], f_x1_part[0], f_x1_part[1]]
    let j_c_0 = f_t_x2_part[0];
    let j_c_1 = f_t_x2_part[1];
    let j_c_2 = f_x1_part[0];
    let j_c_3 = f_x1_part[1];
    let n_j_c = (j_c_0 * j_c_0 + j_c_1 * j_c_1 + j_c_2 * j_c_2 + j_c_3 * j_c_3).sqrt();

    if n_j_c < 1e-10 {
        return 0.0;
    }

    // Sampson error: r = C / ||J_C||
    let inv_n_j_c = 1.0 / n_j_c;
    c * inv_n_j_c
}

/// Reprojection error for absolute pose (OnP).
/// Computes the 2D reprojection error: ||project(R*X + t) - x||
pub fn reprojection_error(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    x_2d: &Vector2<f64>,
    x_3d: &Vector3<f64>,
) -> f64 {
    // Transform 3D point: R*X + t
    let p = r * x_3d + t;

    // Check if point is behind camera
    if p.z <= 0.0 {
        return 1e10; // Large penalty
    }

    // Project to 2D (assuming normalized coordinates, identity intrinsics)
    let projected = Vector2::new(p.x / p.z, p.y / p.z);

    // Compute error
    (projected - x_2d).norm()
}

/// Fundamental matrix parameterization using SVD factorization (Bartoli & Sturm).
/// Maintains rank-2 constraint: F = U * diag(1, sigma, 0) * V^T
/// Parameters: 7 (quaternion for U, quaternion for V, sigma)
#[derive(Clone)]
pub struct FactorizedFundamentalMatrix {
    q_u: UnitQuaternion<f64>,
    q_v: UnitQuaternion<f64>,
    sigma: f64,
}

impl FactorizedFundamentalMatrix {
    pub fn from_fundamental(f: &Matrix3<f64>) -> Self {
        let svd = nalgebra::SVD::new(*f, true, true);
        let mut u = svd.u.unwrap();
        let s = svd.singular_values;
        let mut v = svd.v_t.unwrap().transpose();

        // Ensure positive determinants
        if u.determinant() < 0.0 {
            u = -u;
        }
        if v.determinant() < 0.0 {
            v = -v;
        }

        // Convert to quaternions
        let q_u =
            UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(u));
        let q_v =
            UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(v));

        // Normalize: use s[1]/s[0] as sigma, set s[0] = 1
        let sigma = if s[0] > 1e-10 { s[1] / s[0] } else { 1.0 };

        Self { q_u, q_v, sigma }
    }

    pub fn to_fundamental(&self) -> Matrix3<f64> {
        let rot_u = self.q_u.to_rotation_matrix();
        let rot_v = self.q_v.to_rotation_matrix();
        let u = rot_u.matrix();
        let v = rot_v.matrix();

        // F = U * diag(1, sigma, 0) * V^T = u.col(0) * v.col(0)^T + sigma * u.col(1) * v.col(1)^T
        let u_col0 = u.column(0);
        let u_col1 = u.column(1);
        let v_col0 = v.column(0);
        let v_col1 = v.column(1);
        u_col0 * v_col0.transpose() + self.sigma * u_col1 * v_col1.transpose()
    }

    /// Get parameters for optimization (7 parameters)
    pub fn to_params(&self) -> DVector<f64> {
        let mut params = DVector::<f64>::zeros(7);
        // Quaternion for U (3 params: imaginary part, assuming real part is positive)
        let q_u_imag = self.q_u.quaternion().imag();
        params[0] = q_u_imag.x;
        params[1] = q_u_imag.y;
        params[2] = q_u_imag.z;
        // Quaternion for V (3 params)
        let q_v_imag = self.q_v.quaternion().imag();
        params[3] = q_v_imag.x;
        params[4] = q_v_imag.y;
        params[5] = q_v_imag.z;
        // Sigma
        params[6] = self.sigma;
        params
    }

    pub fn from_params(&mut self, params: &DVector<f64>) {
        if params.len() < 7 {
            return;
        }

        // Reconstruct quaternions from imaginary parts (assuming real part is positive)
        let q_u_imag = Vector3::new(params[0], params[1], params[2]);
        let q_u_w = (1.0 - q_u_imag.norm_squared()).max(0.0).sqrt();
        let q_u = Quaternion::new(q_u_w, q_u_imag.x, q_u_imag.y, q_u_imag.z);
        self.q_u = UnitQuaternion::from_quaternion(q_u.normalize());

        let q_v_imag = Vector3::new(params[3], params[4], params[5]);
        let q_v_w = (1.0 - q_v_imag.norm_squared()).max(0.0).sqrt();
        let q_v = Quaternion::new(q_v_w, q_v_imag.x, q_v_imag.y, q_v_imag.z);
        self.q_v = UnitQuaternion::from_quaternion(q_v.normalize());

        self.sigma = params[6];
    }
}

/// Cost function for fundamental matrix bundle adjustment.
pub struct FundamentalCostFunction {
    x1: Vec<Vector2<f64>>,
    x2: Vec<Vector2<f64>>,
    weights: Option<Vec<f64>>,
}

impl FundamentalCostFunction {
    pub fn new(x1: Vec<Vector2<f64>>, x2: Vec<Vector2<f64>>, weights: Option<Vec<f64>>) -> Self {
        Self { x1, x2, weights }
    }
}

impl CostFunction for FundamentalCostFunction {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // Reconstruct fundamental matrix from parameters
        let mut f_factorized = FactorizedFundamentalMatrix {
            q_u: UnitQuaternion::identity(),
            q_v: UnitQuaternion::identity(),
            sigma: 1.0,
        };
        f_factorized.from_params(param);
        let f = f_factorized.to_fundamental();

        let mut total_cost = 0.0;
        for i in 0..self.x1.len() {
            let error = sampson_error(&f, &self.x1[i], &self.x2[i]);
            let weight = self
                .weights
                .as_ref()
                .and_then(|w| w.get(i))
                .copied()
                .unwrap_or(1.0);
            total_cost += weight * error * error;
        }

        Ok(total_cost)
    }
}

impl Gradient for FundamentalCostFunction {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // Compute numerical gradient
        let mut grad = DVector::<f64>::zeros(param.len());
        let eps = 1e-8;
        let base_cost = self.cost(param)?;

        for i in 0..param.len() {
            let mut param_plus = param.clone();
            param_plus[i] += eps;
            let cost_plus = self.cost(&param_plus)?;
            grad[i] = (cost_plus - base_cost) / eps;
        }

        Ok(grad)
    }
}

/// Refine fundamental matrix using Levenberg-Marquardt optimization.
pub fn refine_fundamental(
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    f: &mut Matrix3<f64>,
    weights: Option<&[f64]>,
    max_iterations: usize,
) -> bool {
    if x1.len() != x2.len() || x1.len() < 8 {
        return false;
    }

    // Convert to Vec<Vector2>
    let x1_vec: Vec<Vector2<f64>> = x1.to_vec();
    let x2_vec: Vec<Vector2<f64>> = x2.to_vec();
    let weights_vec = weights.map(|w| w.to_vec());

    // Initialize parameterization
    let f_factorized = FactorizedFundamentalMatrix::from_fundamental(f);
    let params = f_factorized.to_params();

    // Create cost function
    let cost = FundamentalCostFunction::new(x1_vec, x2_vec, weights_vec);

    // Use gradient descent with line search for optimization
    let alpha = 0.01;
    let mut current_params = params;

    for _iter in 0..max_iterations {
        let grad_result = cost.gradient(&current_params);
        match grad_result {
            Ok(grad) => {
                if grad.norm() < 1e-6 {
                    break;
                }
                current_params = &current_params - &(alpha * &grad);
            }
            Err(_) => break,
        }
    }

    // Reconstruct fundamental matrix
    let mut f_factorized = FactorizedFundamentalMatrix {
        q_u: UnitQuaternion::identity(),
        q_v: UnitQuaternion::identity(),
        sigma: 1.0,
    };
    f_factorized.from_params(&current_params);
    *f = f_factorized.to_fundamental();
    true
}

/// Cost function for absolute pose (OnP) bundle adjustment.
pub struct AbsolutePoseCostFunction {
    points_2d: Vec<Vector2<f64>>,
    points_3d: Vec<Vector3<f64>>,
    weights: Option<Vec<f64>>,
}

impl AbsolutePoseCostFunction {
    pub fn new(
        points_2d: Vec<Vector2<f64>>,
        points_3d: Vec<Vector3<f64>>,
        weights: Option<Vec<f64>>,
    ) -> Self {
        Self {
            points_2d,
            points_3d,
            weights,
        }
    }
}

impl CostFunction for AbsolutePoseCostFunction {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        if param.len() < 6 {
            return Ok(f64::INFINITY);
        }

        // Reconstruct rotation and translation from parameters
        // Parameters: [rx, ry, rz, tx, ty, tz] (rotation as axis-angle, translation)
        let axis_angle = Vector3::new(param[0], param[1], param[2]);
        let angle = axis_angle.norm();
        let r = if angle < 1e-10 {
            Matrix3::identity()
        } else {
            let axis = axis_angle / angle;
            let k = nalgebra::Matrix3::new(
                0.0, -axis.z, axis.y, axis.z, 0.0, -axis.x, -axis.y, axis.x, 0.0,
            );
            Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * (k * k)
        };
        let t = Vector3::new(param[3], param[4], param[5]);

        let mut total_cost = 0.0;
        for i in 0..self.points_2d.len() {
            let error = reprojection_error(&r, &t, &self.points_2d[i], &self.points_3d[i]);
            let weight = self
                .weights
                .as_ref()
                .and_then(|w| w.get(i))
                .copied()
                .unwrap_or(1.0);
            total_cost += weight * error * error;
        }

        Ok(total_cost)
    }
}

impl Gradient for AbsolutePoseCostFunction {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // Compute numerical gradient
        let mut grad = DVector::<f64>::zeros(param.len());
        let eps = 1e-8;
        let base_cost = self.cost(param)?;

        for i in 0..param.len() {
            let mut param_plus = param.clone();
            param_plus[i] += eps;
            let cost_plus = self.cost(&param_plus)?;
            grad[i] = (cost_plus - base_cost) / eps;
        }

        Ok(grad)
    }
}

/// Refine absolute pose using Levenberg-Marquardt optimization.
pub fn refine_absolute_pose(
    points_2d: &[Vector2<f64>],
    points_3d: &[Vector3<f64>],
    r: &mut Matrix3<f64>,
    t: &mut Vector3<f64>,
    weights: Option<&[f64]>,
    max_iterations: usize,
) -> bool {
    if points_2d.len() != points_3d.len() || points_2d.len() < 3 {
        return false;
    }

    // Convert to Vec
    let points_2d_vec: Vec<Vector2<f64>> = points_2d.to_vec();
    let points_3d_vec: Vec<Vector3<f64>> = points_3d.to_vec();
    let weights_vec = weights.map(|w| w.to_vec());

    // Initialize parameters from current pose
    // Convert rotation to axis-angle
    let r_rot = nalgebra::Rotation3::from_matrix_unchecked(*r);
    let axis_angle = r_rot.axis_angle();
    let mut params = DVector::<f64>::zeros(6);
    if let Some((axis, angle)) = axis_angle {
        params[0] = axis.x * angle;
        params[1] = axis.y * angle;
        params[2] = axis.z * angle;
    }
    params[3] = t.x;
    params[4] = t.y;
    params[5] = t.z;

    // Create cost function
    let cost = AbsolutePoseCostFunction::new(points_2d_vec, points_3d_vec, weights_vec);

    // Use gradient descent for optimization
    let alpha = 0.01;
    let mut current_params = params;

    for _iter in 0..max_iterations {
        let grad_result = cost.gradient(&current_params);
        match grad_result {
            Ok(grad) => {
                if grad.norm() < 1e-6 {
                    break;
                }
                current_params = &current_params - &(alpha * &grad);
            }
            Err(_) => break,
        }
    }

    let optimized_params = current_params;

    // Reconstruct rotation and translation
    let axis_angle = Vector3::new(
        optimized_params[0],
        optimized_params[1],
        optimized_params[2],
    );
    let angle = axis_angle.norm();
    if angle > 1e-10 {
        let axis = axis_angle / angle;
        let k = nalgebra::Matrix3::new(
            0.0, -axis.z, axis.y, axis.z, 0.0, -axis.x, -axis.y, axis.x, 0.0,
        );
        *r = Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * (k * k);
    }
    *t = Vector3::new(
        optimized_params[3],
        optimized_params[4],
        optimized_params[5],
    );
    true
}
