//! Fundamental matrix estimator using 8-point and 7-point algorithms.

use crate::core::Estimator;
use crate::models::FundamentalMatrix;
use crate::types::DataMatrix;
use crate::utils::solve_cubic_real;
use nalgebra::{DMatrix, Matrix3, SVD};

pub(crate) fn has_non_collinear_2d_sample(
    data: &DataMatrix,
    sample: &[usize],
    first_column: usize,
) -> bool {
    if sample.len() < 3 || data.n_dims() < first_column + 2 {
        return false;
    }

    let origin_index = sample[0];
    if origin_index >= data.n_points() {
        return false;
    }
    let origin_x = data.get(origin_index, first_column);
    let origin_y = data.get(origin_index, first_column + 1);
    if !origin_x.is_finite() || !origin_y.is_finite() {
        return false;
    }

    let mut direction = (0.0, 0.0);
    let mut scale_squared = 0.0_f64;
    for &index in &sample[1..] {
        if index >= data.n_points() {
            return false;
        }
        let dx = data.get(index, first_column) - origin_x;
        let dy = data.get(index, first_column + 1) - origin_y;
        let distance_squared = dx.mul_add(dx, dy * dy);
        if !distance_squared.is_finite() {
            return false;
        }
        if distance_squared > scale_squared {
            direction = (dx, dy);
            scale_squared = distance_squared;
        }
    }

    if scale_squared < 1e-24 {
        return false;
    }

    sample[1..].iter().any(|&index| {
        let dx = data.get(index, first_column) - origin_x;
        let dy = data.get(index, first_column + 1) - origin_y;
        (direction.0 * dy - direction.1 * dx).abs() > 1e-10 * scale_squared
    })
}

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

    /// Project an eight-point estimate onto the rank-two fundamental-matrix manifold.
    fn enforce_rank_two(&self, matrix: Matrix3<f64>) -> Matrix3<f64> {
        let svd = SVD::new(matrix, true, true);
        let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
            return matrix;
        };
        let mut singular_values = svd.singular_values;
        singular_values[2] = 0.0;
        u * Matrix3::from_diagonal(&singular_values) * v_t
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
            cx1 += data.get(idx, 0);
            cy1 += data.get(idx, 1);
            cx2 += data.get(idx, 2);
            cy2 += data.get(idx, 3);
        }

        cx1 /= n as f64;
        cy1 /= n as f64;
        cx2 /= n as f64;
        cy2 /= n as f64;

        // Compute mean distances
        let mut d1 = 0.0;
        let mut d2 = 0.0;

        for &idx in sample {
            let dx1 = data.get(idx, 0) - cx1;
            let dy1 = data.get(idx, 1) - cy1;
            let dx2 = data.get(idx, 2) - cx2;
            let dy2 = data.get(idx, 3) - cy2;
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
            normalized[(i, 0)] = (data.get(idx, 0) - cx1) * s1;
            normalized[(i, 1)] = (data.get(idx, 1) - cy1) * s1;
            normalized[(i, 2)] = (data.get(idx, 2) - cx2) * s2;
            normalized[(i, 3)] = (data.get(idx, 3) - cy2) * s2;
        }

        true
    }

    /// Seven-point fundamental matrix solver.
    ///
    /// Correspondences are normalized before finding the two-dimensional null
    /// space. This avoids the scale sensitivity of solving the cubic directly
    /// in pixel coordinates.
    fn estimate_seven_point(&self, data: &DataMatrix, sample: &[usize]) -> Vec<FundamentalMatrix> {
        let mut normalized = DMatrix::<f64>::zeros(0, 0);
        let mut t1 = Matrix3::<f64>::zeros();
        let mut t2 = Matrix3::<f64>::zeros();
        if !self.normalize_points(data, sample, &mut normalized, &mut t1, &mut t2) {
            return Vec::new();
        }

        // Pad the seven equations to a square matrix. Nalgebra's rectangular
        // SVD exposes only a thin V^T, while the square decomposition retains
        // both null-space vectors needed by the seven-point algorithm.
        let mut coefficients = DMatrix::<f64>::zeros(9, 9);
        for i in 0..7 {
            let x0 = normalized[(i, 0)];
            let y0 = normalized[(i, 1)];
            let x1 = normalized[(i, 2)];
            let y1 = normalized[(i, 3)];

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
        let svd = SVD::new(coefficients, false, true);
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

        let matrix_from_vector =
            |vector: nalgebra::DVectorView<'_, f64>| Matrix3::from_row_slice(vector.as_slice());
        let f1 = matrix_from_vector(f1);
        let f2 = matrix_from_vector(f2);
        // The two null-space basis matrices are arbitrary. Choose the one
        // with the larger determinant as the cubic direction so its leading
        // coefficient is not spuriously zero.
        let (f1, f2) = if f1.determinant().abs() >= f2.determinant().abs() {
            (f1, f2)
        } else {
            (f2, f1)
        };

        // det(lambda * f1 + f2) is cubic. Evaluating it at -1, 0, 1, and 2
        // avoids error-prone hand indexing of the null-space basis.
        let determinant = |lambda: f64| (lambda * f1 + f2).determinant();
        let d_minus_one = determinant(-1.0);
        let c0 = determinant(0.0);
        let d_one = determinant(1.0);
        let d_two = determinant(2.0);
        let c2 = (d_one + d_minus_one) * 0.5 - c0;
        let c1_plus_c3 = (d_one - d_minus_one) * 0.5;
        let c3 = (d_two - c0 - 4.0 * c2 - 2.0 * c1_plus_c3) / 6.0;
        let c1 = c1_plus_c3 - c3;

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
            let f = t2.transpose() * (lambda * f1 + f2) * t1;
            let norm = f.norm();
            if norm > 1e-10 {
                models.push(FundamentalMatrix::new(self.enforce_rank_two(f / norm)));
            }
        }

        models
    }
}

impl Estimator for FundamentalEstimator {
    type Model = FundamentalMatrix;

    fn sample_size(&self) -> usize {
        7
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
        }) && has_non_collinear_2d_sample(data, sample, 0)
            && has_non_collinear_2d_sample(data, sample, 2)
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

        // Solve A f = 0 directly. Forming A^T A would square the condition
        // number and materially degrades estimates on difficult image pairs.
        let a = if a.nrows() < a.ncols() {
            let mut padded = DMatrix::<f64>::zeros(a.ncols(), a.ncols());
            padded.rows_mut(0, a.nrows()).copy_from(&a);
            padded
        } else {
            a
        };
        let svd = SVD::new(a, false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let last_row = vt.row(vt.nrows() - 1);

        // Check for NaN
        if last_row.iter().any(|&x| !x.is_finite()) {
            return Vec::new();
        }

        // Reshape into 3x3 matrix
        let mut f_norm = Matrix3::<f64>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                f_norm[(r, c)] = last_row[3 * r + c];
            }
        }

        // Denormalize: F = T2^T * F_norm * T1
        let f = t2.transpose() * self.enforce_rank_two(f_norm) * t1;
        let norm = f.norm();
        if !norm.is_finite() || norm < 1e-12 {
            return Vec::new();
        }

        vec![FundamentalMatrix::new(f / norm)]
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

        // Solve A f = 0 directly to preserve the conditioning gained from
        // Hartley normalization.
        let a = if a.nrows() < a.ncols() {
            let mut padded = DMatrix::<f64>::zeros(a.ncols(), a.ncols());
            padded.rows_mut(0, a.nrows()).copy_from(&a);
            padded
        } else {
            a
        };
        let svd = SVD::new(a, false, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => return Vec::new(),
        };
        let last_row = vt.row(vt.nrows() - 1);

        // Check for NaN
        if last_row.iter().any(|&x| !x.is_finite()) {
            return Vec::new();
        }

        // Reshape into 3x3 matrix
        let mut f_norm = Matrix3::<f64>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                f_norm[(r, c)] = last_row[3 * r + c];
            }
        }

        // Denormalize: F = T2^T * F_norm * T1
        let f = t2.transpose() * self.enforce_rank_two(f_norm) * t1;
        let norm = f.norm();
        if !norm.is_finite() || norm < 1e-12 {
            return Vec::new();
        }

        vec![FundamentalMatrix::new(f / norm)]
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        let norm = model.f.norm();
        norm.is_finite()
            && norm > 1e-12
            && model.f.iter().all(|value| value.is_finite())
            // det(F) scales cubically with F, so compare a normalized rank
            // residual instead of an absolute determinant.
            && model.f.determinant().abs() / norm.powi(3) < 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FundamentalMatrix;

    fn calibrated_translation_correspondences(
        count: usize,
        pixel_scale: f64,
        offset: (f64, f64),
    ) -> DataMatrix {
        let points = [
            (-0.7, -0.4, 2.1),
            (0.2, -0.6, 2.6),
            (0.8, -0.2, 3.4),
            (-0.5, 0.3, 2.8),
            (0.1, 0.5, 3.1),
            (0.6, 0.7, 2.4),
            (-0.3, 0.8, 3.7),
            (0.9, 0.1, 4.2),
            (-0.8, 0.6, 3.3),
        ];
        let translation = (0.35, -0.12, 0.18);
        let (sin_angle, cos_angle) = 0.27_f64.sin_cos();
        let mut data = DataMatrix::zeros(count, 4);
        for (index, &(x, y, z)) in points.iter().take(count).enumerate() {
            data.set(index, 0, offset.0 + pixel_scale * x / z);
            data.set(index, 1, offset.1 + pixel_scale * y / z);
            let rotated_x = cos_angle * x + sin_angle * z;
            let rotated_z = -sin_angle * x + cos_angle * z;
            data.set(
                index,
                2,
                offset.0 + pixel_scale * (rotated_x + translation.0) / (rotated_z + translation.2),
            );
            data.set(
                index,
                3,
                offset.1 + pixel_scale * (y + translation.1) / (rotated_z + translation.2),
            );
        }
        data
    }

    fn mean_sampson_error(data: &DataMatrix, model: &FundamentalMatrix) -> f64 {
        (0..data.n_points())
            .map(|index| {
                crate::bundle_adjustment::sampson_error(
                    &model.f,
                    &nalgebra::Vector2::new(data.get(index, 0), data.get(index, 1)),
                    &nalgebra::Vector2::new(data.get(index, 2), data.get(index, 3)),
                )
            })
            .sum::<f64>()
            / data.n_points() as f64
    }

    #[test]
    fn rank_two_projection_removes_the_smallest_singular_value() {
        let estimator = FundamentalEstimator::new();
        let matrix = Matrix3::new(1.0, 2.0, 3.0, 0.5, -1.0, 4.0, 2.0, 0.25, 1.0);
        let projected = estimator.enforce_rank_two(matrix);

        assert!(projected.determinant().abs() < 1e-12);
    }

    #[test]
    fn model_validation_rejects_the_zero_matrix() {
        let estimator = FundamentalEstimator::new();
        assert!(!estimator.is_valid_model(
            &FundamentalMatrix::new(Matrix3::zeros()),
            &DataMatrix::zeros(0, 4),
            &[],
            1.0,
        ));
    }

    #[test]
    fn sample_validation_rejects_collinear_image_points() {
        let estimator = FundamentalEstimator::new();
        let mut data = DataMatrix::zeros(7, 4);
        for row in 0..7 {
            let coordinate = row as f64;
            data.set(row, 0, coordinate);
            data.set(row, 1, 2.0 * coordinate);
            data.set(row, 2, coordinate + 1.0);
            data.set(row, 3, 2.0 * coordinate - 1.0);
        }

        assert!(!estimator.is_valid_sample(&data, &[0, 1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn seven_point_minimal_solver_recovers_an_exact_epipolar_model() {
        let estimator = FundamentalEstimator::new();
        let data = calibrated_translation_correspondences(7, 1.0, (0.0, 0.0));
        let sample: Vec<_> = (0..7).collect();

        assert_eq!(estimator.sample_size(), 7);
        assert!(estimator.is_valid_sample(&data, &sample));

        let models = estimator.estimate_model(&data, &sample);
        assert!(
            models
                .iter()
                .any(|model| mean_sampson_error(&data, model) < 1e-12),
            "seven-point solver errors: {:?}",
            models
                .iter()
                .map(|model| mean_sampson_error(&data, model))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn eight_point_fit_is_stable_for_large_pixel_coordinates() {
        let estimator = FundamentalEstimator::new();
        let data = calibrated_translation_correspondences(9, 4_000.0, (1.0e7, -8.0e6));
        let sample: Vec<_> = (0..9).collect();

        let models = estimator.estimate_model(&data, &sample);
        let model = models
            .first()
            .expect("eight-point solver should produce a model");
        assert!(model.f.iter().all(|value| value.is_finite()));
        let error = mean_sampson_error(&data, model);
        assert!(
            error < 1e-5,
            "large-coordinate fit should retain a small Sampson error, got {error}"
        );
    }
}
