//! Fundamental matrix estimator using 8-point and 7-point algorithms.

use crate::core::Estimator;
use crate::models::FundamentalMatrix;
use crate::types::DataMatrix;
use crate::utils::solve_cubic_real;
use nalgebra::{DMatrix, DVector, Matrix3, SVD};

/// Fundamental matrix estimator using the 8-point algorithm.
pub struct FundamentalEstimator;

impl Default for FundamentalEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl FundamentalEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Normalize points for numerical stability (Hartley normalization).
    fn normalize_points(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        normalized: &mut DMatrix<f64>,
        t1: &mut Matrix3<f64>,
        t2: &mut Matrix3<f64>,
    ) -> bool {
        let n = sample.len();
        if n == 0 {
            return false;
        }

        // Compute centroids
        let mut cx1 = 0.0;
        let mut cy1 = 0.0;
        let mut cx2 = 0.0;
        let mut cy2 = 0.0;

        for &idx in sample {
            cx1 += data[(idx, 0)];
            cy1 += data[(idx, 1)];
            cx2 += data[(idx, 2)];
            cy2 += data[(idx, 3)];
        }

        cx1 /= n as f64;
        cy1 /= n as f64;
        cx2 /= n as f64;
        cy2 /= n as f64;

        // Compute mean distances
        let mut d1 = 0.0;
        let mut d2 = 0.0;

        for &idx in sample {
            let dx1 = data[(idx, 0)] - cx1;
            let dy1 = data[(idx, 1)] - cy1;
            let dx2 = data[(idx, 2)] - cx2;
            let dy2 = data[(idx, 3)] - cy2;
            d1 += (dx1 * dx1 + dy1 * dy1).sqrt();
            d2 += (dx2 * dx2 + dy2 * dy2).sqrt();
        }

        d1 /= n as f64;
        d2 /= n as f64;

        if d1 < 1e-10 || d2 < 1e-10 {
            return false;
        }

        let s1 = std::f64::consts::SQRT_2 / d1;
        let s2 = std::f64::consts::SQRT_2 / d2;

        // Build normalization transforms
        *t1 = Matrix3::new(s1, 0.0, -s1 * cx1, 0.0, s1, -s1 * cy1, 0.0, 0.0, 1.0);
        *t2 = Matrix3::new(s2, 0.0, -s2 * cx2, 0.0, s2, -s2 * cy2, 0.0, 0.0, 1.0);

        // Normalize points
        normalized.resize_mut(n, 4, 0.0);
        for (i, &idx) in sample.iter().enumerate() {
            normalized[(i, 0)] = (data[(idx, 0)] - cx1) * s1;
            normalized[(i, 1)] = (data[(idx, 1)] - cy1) * s1;
            normalized[(i, 2)] = (data[(idx, 2)] - cx2) * s2;
            normalized[(i, 3)] = (data[(idx, 3)] - cy2) * s2;
        }

        true
    }

    /// Seven-point fundamental matrix solver (matches C++ implementation).
    /// Returns up to 3 solutions by solving a cubic equation.
    fn estimate_seven_point(&self, data: &DataMatrix, sample: &[usize]) -> Vec<FundamentalMatrix> {
        // Build 7x9 coefficient matrix
        let mut coefficients = DMatrix::<f64>::zeros(7, 9);
        for (i, &idx) in sample.iter().enumerate() {
            let x0 = data[(idx, 0)];
            let y0 = data[(idx, 1)];
            let x1 = data[(idx, 2)];
            let y1 = data[(idx, 3)];

            coefficients[(i, 0)] = x1 * x0;
            coefficients[(i, 1)] = x1 * y0;
            coefficients[(i, 2)] = x1;
            coefficients[(i, 3)] = y1 * x0;
            coefficients[(i, 4)] = y1 * y0;
            coefficients[(i, 5)] = y1;
            coefficients[(i, 6)] = x0;
            coefficients[(i, 7)] = y0;
            coefficients[(i, 8)] = 1.0;
        }

        // Find null space using SVD (last 2 columns of V correspond to null space)
        let svd = SVD::new(coefficients.clone(), false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let v = vt.transpose();

        // Last 2 columns are the null space basis
        if v.ncols() < 9 {
            return Vec::new();
        }

        // Extract f1 and f2 from null space (columns 7 and 8)
        let f1 = v.column(7);
        let f2 = v.column(8);

        // Compute cubic polynomial coefficients for det(lambda*f1 + mu*f2) = 0
        // where we set mu = 1 and solve for lambda
        // The determinant is a cubic in lambda
        let mut c3 = 0.0;
        let mut c2 = 0.0;
        let mut c1 = 0.0;
        let mut c0 = 0.0;

        // Compute determinant coefficients manually (matching C++ implementation)
        // det(F) = F[0]*(F[4]*F[8] - F[5]*F[7]) - F[1]*(F[3]*F[8] - F[5]*F[6]) + F[2]*(F[3]*F[7] - F[4]*F[6])
        // f1 and f2 are stored in column-major order: [f00, f10, f20, f01, f11, f21, f02, f12, f22]
        let idx = |r: usize, c: usize| c * 3 + r;

        let f1_0 = f1[idx(0, 0)];
        let f1_1 = f1[idx(0, 1)];
        let f1_2 = f1[idx(0, 2)];
        let f1_3 = f1[idx(1, 0)];
        let f1_4 = f1[idx(1, 1)];
        let f1_5 = f1[idx(1, 2)];
        let f1_6 = f1[idx(2, 0)];
        let f1_7 = f1[idx(2, 1)];
        let f1_8 = f1[idx(2, 2)];

        let f2_0 = f2[idx(0, 0)];
        let f2_1 = f2[idx(0, 1)];
        let f2_2 = f2[idx(0, 2)];
        let f2_3 = f2[idx(1, 0)];
        let f2_4 = f2[idx(1, 1)];
        let f2_5 = f2[idx(1, 2)];
        let f2_6 = f2[idx(2, 0)];
        let f2_7 = f2[idx(2, 1)];
        let f2_8 = f2[idx(2, 2)];

        // Compute cubic coefficients (matching C++ code exactly)
        c3 = f1_0 * (f1_4 * f1_8 - f1_5 * f1_7) - f1_1 * (f1_3 * f1_8 - f1_5 * f1_6)
            + f1_2 * (f1_3 * f1_7 - f1_4 * f1_6);

        c2 = f1_0 * (f1_4 * f2_8 + f2_4 * f1_8 - f1_5 * f2_7 - f2_5 * f1_7)
            + f2_0 * (f1_4 * f1_8 - f1_5 * f1_7)
            - f1_1 * (f1_3 * f2_8 + f2_3 * f1_8 - f1_5 * f2_6 - f2_5 * f1_6)
            - f2_1 * (f1_3 * f1_8 - f1_5 * f1_6)
            + f1_2 * (f1_3 * f2_7 + f2_3 * f1_7 - f1_4 * f2_6 - f2_4 * f1_6)
            + f2_2 * (f1_3 * f1_7 - f1_4 * f1_6);

        c1 = f1_0 * (f2_4 * f2_8 - f2_5 * f2_7)
            + f2_0 * (f1_4 * f2_8 + f2_4 * f1_8 - f1_5 * f2_7 - f2_5 * f1_7)
            - f1_1 * (f2_3 * f2_8 - f2_5 * f2_6)
            - f2_1 * (f1_3 * f2_8 + f2_3 * f1_8 - f1_5 * f2_6 - f2_5 * f1_6)
            + f1_2 * (f2_3 * f2_7 - f2_4 * f2_6)
            + f2_2 * (f1_3 * f2_7 + f2_3 * f1_7 - f1_4 * f2_6 - f2_4 * f1_6);

        c0 = f2_0 * (f2_4 * f2_8 - f2_5 * f2_7) - f2_1 * (f2_3 * f2_8 - f2_5 * f2_6)
            + f2_2 * (f2_3 * f2_7 - f2_4 * f2_6);

        // Normalize polynomial (divide by c3 to get monic form)
        if c3.abs() < 1e-10_f64 {
            return Vec::new();
        }
        let inv_c3 = 1.0 / c3;
        let c2_norm = c2 * inv_c3;
        let c1_norm = c1 * inv_c3;
        let c0_norm = c0 * inv_c3;

        // Solve cubic: x^3 + c2*x^2 + c1*x + c0 = 0
        let mut roots = [0.0; 3];
        let n_roots = solve_cubic_real(c2_norm, c1_norm, c0_norm, &mut roots);

        // Build fundamental matrices for each root
        let mut models = Vec::new();
        for root in roots.iter().take(n_roots) {
            let lambda = *root;
            // F = lambda * f1 + f2
            let mut f_vec = DVector::<f64>::zeros(9);
            for j in 0..9 {
                f_vec[j] = lambda * f1[j] + f2[j];
            }

            // Normalize
            let norm = f_vec.norm();
            if norm > 1e-10 {
                f_vec /= norm;
            }

            // Reshape into 3x3 matrix
            let mut f = Matrix3::<f64>::zeros();
            for r in 0..3 {
                for c in 0..3 {
                    f[(r, c)] = f_vec[r * 3 + c];
                }
            }

            models.push(FundamentalMatrix::new(f));
        }

        models
    }
}

impl Estimator for FundamentalEstimator {
    type Model = FundamentalMatrix;

    fn sample_size(&self) -> usize {
        8
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
        if n < 7 {
            return Vec::new();
        }

        // Use 7-point solver for exactly 7 points, 8-point for 8+ points
        if n == 7 {
            return self.estimate_seven_point(data, sample);
        }

        // 8-point solver for 8+ points
        // Normalize points
        let mut normalized = DMatrix::<f64>::zeros(0, 0);
        let mut t1 = Matrix3::<f64>::zeros();
        let mut t2 = Matrix3::<f64>::zeros();

        if !self.normalize_points(data, sample, &mut normalized, &mut t1, &mut t2) {
            return Vec::new();
        }

        // Build coefficient matrix A for Af = 0
        let mut a = DMatrix::<f64>::zeros(n, 9);
        for i in 0..n {
            let x1 = normalized[(i, 0)];
            let y1 = normalized[(i, 1)];
            let x2 = normalized[(i, 2)];
            let y2 = normalized[(i, 3)];

            a[(i, 0)] = x2 * x1;
            a[(i, 1)] = x2 * y1;
            a[(i, 2)] = x2;
            a[(i, 3)] = y2 * x1;
            a[(i, 4)] = y2 * y1;
            a[(i, 5)] = y2;
            a[(i, 6)] = x1;
            a[(i, 7)] = y1;
            a[(i, 8)] = 1.0;
        }

        // Solve via SVD: the solution is the right singular vector
        // corresponding to the smallest singular value of A^T A
        let ata = &a.transpose() * &a;
        let svd = SVD::new(ata, false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let v = vt.transpose();

        // Last column corresponds to smallest singular value
        let last_col = v.column(v.ncols() - 1);

        // Check for NaN
        if last_col.iter().any(|&x| x.is_nan()) {
            return Vec::new();
        }

        // Reshape into 3x3 matrix
        let mut f_norm = Matrix3::<f64>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                f_norm[(r, c)] = last_col[3 * r + c];
            }
        }

        // Denormalize: F = T2^T * F_norm * T1
        let f = t2.transpose() * f_norm * t1;

        vec![FundamentalMatrix::new(f)]
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        let n = sample.len();
        if n < self.sample_size() {
            return Vec::new();
        }

        // Normalize points
        let mut normalized = DMatrix::<f64>::zeros(0, 0);
        let mut t1 = Matrix3::<f64>::zeros();
        let mut t2 = Matrix3::<f64>::zeros();

        if !self.normalize_points(data, sample, &mut normalized, &mut t1, &mut t2) {
            return Vec::new();
        }

        // Build coefficient matrix A for Af = 0 (with optional weights)
        let mut a = DMatrix::<f64>::zeros(n, 9);
        for (i, &idx) in sample.iter().enumerate() {
            let x1 = normalized[(i, 0)];
            let y1 = normalized[(i, 1)];
            let x2 = normalized[(i, 2)];
            let y2 = normalized[(i, 3)];

            let weight = weights.map(|w| w[idx]).unwrap_or(1.0);
            let w_x0 = weight * x1;
            let w_y0 = weight * y1;
            let w_x1 = weight * x2;
            let w_y1 = weight * y2;

            a[(i, 0)] = w_x1 * x1;
            a[(i, 1)] = w_x1 * y1;
            a[(i, 2)] = w_x1;
            a[(i, 3)] = w_y1 * x1;
            a[(i, 4)] = w_y1 * y1;
            a[(i, 5)] = w_y1;
            a[(i, 6)] = w_x0;
            a[(i, 7)] = w_y0;
            a[(i, 8)] = weight;
        }

        // Solve via SVD: A^T * A * f = 0
        let ata = &a.transpose() * &a;
        let svd = SVD::new(ata, false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let v = vt.transpose();

        // Last column corresponds to smallest singular value
        let last_col = v.column(v.ncols() - 1);

        // Check for NaN
        if last_col.iter().any(|&x| x.is_nan()) {
            return Vec::new();
        }

        // Reshape into 3x3 matrix
        let mut f_norm = Matrix3::<f64>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                f_norm[(r, c)] = last_col[3 * r + c];
            }
        }

        // Denormalize: F = T2^T * F_norm * T1
        let f = t2.transpose() * f_norm * t1;

        vec![FundamentalMatrix::new(f)]
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        // Basic sanity check: determinant should be small (rank-2 constraint)
        let det = model.f.determinant().abs();
        det < 1e-3 * _threshold.max(1.0)
    }
}
