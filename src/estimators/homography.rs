//! Homography estimator using normalized direct linear transformation.

use crate::core::Estimator;
use crate::models::Homography;
use crate::types::DataMatrix;
use nalgebra::{DMatrix, Matrix3, SVD};

/// Homography estimator using the four-point DLT algorithm.
pub struct HomographyEstimator;

impl Default for HomographyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl HomographyEstimator {
    pub fn new() -> Self {
        Self
    }

    fn non_collinear_triplet(
        data: &DataMatrix,
        first: usize,
        second: usize,
        third: usize,
        offset: usize,
    ) -> bool {
        let first = nalgebra::Vector2::new(data.get(first, offset), data.get(first, offset + 1));
        let second = nalgebra::Vector2::new(data.get(second, offset), data.get(second, offset + 1));
        let third = nalgebra::Vector2::new(data.get(third, offset), data.get(third, offset + 1));
        let first_edge = second - first;
        let second_edge = third - first;
        let scale_squared = first_edge.norm_squared().max(second_edge.norm_squared());
        scale_squared.is_finite()
            && scale_squared > 1e-24
            && (first_edge.x * second_edge.y - first_edge.y * second_edge.x).abs()
                > 1e-10 * scale_squared
    }

    /// Apply Hartley normalization independently to both image planes.
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

        let mut cx1 = 0.0;
        let mut cy1 = 0.0;
        let mut cx2 = 0.0;
        let mut cy2 = 0.0;
        for &index in sample {
            cx1 += data.get(index, 0);
            cy1 += data.get(index, 1);
            cx2 += data.get(index, 2);
            cy2 += data.get(index, 3);
        }
        let n = n as f64;
        cx1 /= n;
        cy1 /= n;
        cx2 /= n;
        cy2 /= n;

        let mut distance1 = 0.0;
        let mut distance2 = 0.0;
        for &index in sample {
            let dx1 = data.get(index, 0) - cx1;
            let dy1 = data.get(index, 1) - cy1;
            let dx2 = data.get(index, 2) - cx2;
            let dy2 = data.get(index, 3) - cy2;
            distance1 += (dx1 * dx1 + dy1 * dy1).sqrt();
            distance2 += (dx2 * dx2 + dy2 * dy2).sqrt();
        }
        distance1 /= n;
        distance2 /= n;
        if !distance1.is_finite()
            || !distance2.is_finite()
            || distance1 < 1e-12
            || distance2 < 1e-12
        {
            return false;
        }

        let scale1 = std::f64::consts::SQRT_2 / distance1;
        let scale2 = std::f64::consts::SQRT_2 / distance2;
        *t1 = Matrix3::new(
            scale1,
            0.0,
            -scale1 * cx1,
            0.0,
            scale1,
            -scale1 * cy1,
            0.0,
            0.0,
            1.0,
        );
        *t2 = Matrix3::new(
            scale2,
            0.0,
            -scale2 * cx2,
            0.0,
            scale2,
            -scale2 * cy2,
            0.0,
            0.0,
            1.0,
        );

        normalized.resize_mut(sample.len(), 4, 0.0);
        for (row, &index) in sample.iter().enumerate() {
            normalized[(row, 0)] = (data.get(index, 0) - cx1) * scale1;
            normalized[(row, 1)] = (data.get(index, 1) - cy1) * scale1;
            normalized[(row, 2)] = (data.get(index, 2) - cx2) * scale2;
            normalized[(row, 3)] = (data.get(index, 3) - cy2) * scale2;
        }
        true
    }

    fn solve_dlt(
        &self,
        normalized: &DMatrix<f64>,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Option<Matrix3<f64>> {
        let n = normalized.nrows();
        let mut coefficients = DMatrix::<f64>::zeros(2 * n, 9);
        for row in 0..n {
            let x1 = normalized[(row, 0)];
            let y1 = normalized[(row, 1)];
            let x2 = normalized[(row, 2)];
            let y2 = normalized[(row, 3)];
            let weight = weights.map(|values| values[sample[row]]).unwrap_or(1.0);
            if !weight.is_finite() || weight < 0.0 {
                return None;
            }
            let weight = weight.sqrt();

            coefficients[(2 * row, 0)] = -weight * x1;
            coefficients[(2 * row, 1)] = -weight * y1;
            coefficients[(2 * row, 2)] = -weight;
            coefficients[(2 * row, 6)] = weight * x2 * x1;
            coefficients[(2 * row, 7)] = weight * x2 * y1;
            coefficients[(2 * row, 8)] = weight * x2;

            coefficients[(2 * row + 1, 3)] = -weight * x1;
            coefficients[(2 * row + 1, 4)] = -weight * y1;
            coefficients[(2 * row + 1, 5)] = -weight;
            coefficients[(2 * row + 1, 6)] = weight * y2 * x1;
            coefficients[(2 * row + 1, 7)] = weight * y2 * y1;
            coefficients[(2 * row + 1, 8)] = weight * y2;
        }

        // A four-point minimal sample has only eight equations. Pad it to a
        // square matrix so nalgebra retains the full right null-space vector.
        let coefficients = if coefficients.nrows() < coefficients.ncols() {
            let mut padded = DMatrix::<f64>::zeros(coefficients.ncols(), coefficients.ncols());
            padded
                .rows_mut(0, coefficients.nrows())
                .copy_from(&coefficients);
            padded
        } else {
            coefficients
        };
        let svd = SVD::new(coefficients, false, true);
        let vt = svd.v_t?;
        let solution = vt.row(vt.nrows() - 1);
        if solution.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some(Matrix3::new(
            solution[0],
            solution[1],
            solution[2],
            solution[3],
            solution[4],
            solution[5],
            solution[6],
            solution[7],
            solution[8],
        ))
    }

    fn estimate_normalized(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Homography> {
        let mut normalized = DMatrix::<f64>::zeros(0, 0);
        let mut t1 = Matrix3::<f64>::zeros();
        let mut t2 = Matrix3::<f64>::zeros();
        if !self.normalize_points(data, sample, &mut normalized, &mut t1, &mut t2) {
            return Vec::new();
        }
        let Some(h_normalized) = self.solve_dlt(&normalized, sample, weights) else {
            return Vec::new();
        };
        let Some(inverse_t2) = t2.try_inverse() else {
            return Vec::new();
        };
        let mut h = inverse_t2 * h_normalized * t1;
        let scale = h[(2, 2)];
        if !scale.is_finite() || scale.abs() < 1e-12 {
            return Vec::new();
        }
        h /= scale;
        if h.iter().any(|value| !value.is_finite()) {
            return Vec::new();
        }
        vec![Homography::new(h)]
    }
}

impl Estimator for HomographyEstimator {
    type Model = Homography;

    fn sample_size(&self) -> usize {
        4
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < self.sample_size() || data.n_dims() < 4 {
            return false;
        }
        for i in 0..sample.len() {
            for j in (i + 1)..sample.len() {
                if sample[i] == sample[j] {
                    return false;
                }
            }
        }
        if sample.iter().any(|&index| {
            index >= data.n_points() || (0..4).any(|column| !data.get(index, column).is_finite())
        }) {
            return false;
        }
        for first in 0..sample.len() {
            for second in (first + 1)..sample.len() {
                for third in (second + 1)..sample.len() {
                    if !Self::non_collinear_triplet(
                        data,
                        sample[first],
                        sample[second],
                        sample[third],
                        0,
                    ) || !Self::non_collinear_triplet(
                        data,
                        sample[first],
                        sample[second],
                        sample[third],
                        2,
                    ) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        if sample.len() < self.sample_size() {
            return Vec::new();
        }
        self.estimate_normalized(data, sample, None)
    }

    fn estimate_model_nonminimal(
        &self,
        data: &DataMatrix,
        sample: &[usize],
        weights: Option<&[f64]>,
    ) -> Vec<Self::Model> {
        if sample.len() < self.sample_size() {
            return Vec::new();
        }
        self.estimate_normalized(data, sample, weights)
    }

    fn is_valid_model(
        &self,
        model: &Homography,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        let determinant = model.h.determinant().abs();
        determinant > 1e-4 && determinant < 1e4
    }
}
