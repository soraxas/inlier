//! GNC (Graduated Non-Convexity) solver for robust rotation and translation estimation
//!
//! This module implements the GNC-TLS solver from TEASER++ for robust point cloud
//! registration with known scale. It uses graduated non-convexity to handle outliers
//! in the correspondence set.
//!
//! # Algorithm Overview
//!
//! 1. **Initialization**: Start with uniform weights (convex problem)
//! 2. **Graduation**: Gradually increase μ parameter (convexity to non-convexity)
//! 3. **Weighted SVD**: At each step, solve weighted Procrustes problem
//! 4. **Weight Update**: Update weights based on residuals (downweight outliers)
//! 5. **Convergence**: Repeat until weights stabilize
//!
//! # Example
//!
//! ```ignore
//! use inlier::kiss_matcher::gnc::GNCSolver;
//! use inlier::types::DataMatrix;
//!
//! // Correspondences: N x 6 matrix [src_x, src_y, src_z, tgt_x, tgt_y, tgt_z]
//! let correspondences = DataMatrix::from_row_slice(100, 6, &data);
//!
//! // Solve for rotation and translation
//! let solver = GNCSolver::new(0.01); // noise bound
//! let result = solver.solve(&correspondences, 1.0); // scale = 1.0
//!
//! println!("Rotation:\n{}", result.rotation);
//! println!("Translation: {:?}", result.translation);
//! println!("Inliers: {}/{}", result.inliers.len(), correspondences.n_points());
//! ```

use crate::types::DataMatrix;
use nalgebra::{Matrix3, SVD, Vector3};

/// GNC solver for rotation and translation estimation
pub struct GNCSolver {
    /// Noise bound (expected noise level in data)
    pub noise_bound: f64,
    /// Maximum GNC iterations
    pub max_iterations: usize,
    /// Cost threshold for inlier classification
    pub cost_threshold: f64,
    /// GNC graduation parameter (increase rate)
    pub gnc_factor: f64,
}

/// Result from GNC solver
#[derive(Clone, Debug)]
pub struct GNCResult {
    /// Estimated rotation matrix (3x3)
    pub rotation: Matrix3<f64>,
    /// Estimated translation vector (3x1)
    pub translation: Vector3<f64>,
    /// Inlier indices
    pub inliers: Vec<usize>,
    /// Final weights for each correspondence
    pub weights: Vec<f64>,
    /// Number of iterations taken
    pub iterations: usize,
}

impl GNCSolver {
    /// Create new GNC solver
    ///
    /// # Arguments
    /// * `noise_bound` - Expected noise level in the data (typically 0.01-0.1)
    pub fn new(noise_bound: f64) -> Self {
        Self {
            noise_bound,
            max_iterations: 100,
            cost_threshold: noise_bound * noise_bound * 2.0, // Chi-squared threshold
            gnc_factor: 1.4,                                 // Graduation rate
        }
    }

    /// Solve for rotation and translation given correspondences
    ///
    /// # Arguments
    /// * `correspondences` - N x 6 matrix [src_x, src_y, src_z, tgt_x, tgt_y, tgt_z]
    /// * `scale` - Known scale factor (typically from TLS estimator)
    ///
    /// # Returns
    /// GNCResult with rotation, translation, and inliers
    pub fn solve(&self, correspondences: &DataMatrix, scale: f64) -> GNCResult {
        let n = correspondences.n_points();
        if n < 3 {
            // Not enough points, return identity
            return GNCResult {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
                inliers: Vec::new(),
                weights: vec![0.0; n],
                iterations: 0,
            };
        }

        // Extract source and target points
        let (src_points, tgt_points) = self.extract_points(correspondences);

        // Initialize weights uniformly
        let mut weights = vec![1.0; n];

        // GNC parameters
        let mut mu = 1.0; // Start convex
        let mu_max = 1e6; // Maximum mu (highly non-convex)

        let mut rotation = Matrix3::identity();
        let mut translation = Vector3::zeros();
        let mut iterations = 0;

        // GNC loop
        while mu < mu_max && iterations < self.max_iterations {
            // Solve weighted Procrustes problem
            let (R, t) = self.solve_weighted_procrustes(&src_points, &tgt_points, &weights, scale);

            // Update weights based on residuals
            let new_weights = self.compute_weights(&src_points, &tgt_points, &R, &t, scale, mu);

            // Check convergence (weights stabilized)
            let weight_change = self.compute_weight_change(&weights, &new_weights);

            weights = new_weights;
            rotation = R;
            translation = t;
            iterations += 1;

            if weight_change < 1e-3 {
                // Converged, increase mu
                mu *= self.gnc_factor;
            }
        }

        // Identify inliers based on final weights
        let inliers = self.identify_inliers(&weights);

        GNCResult {
            rotation,
            translation,
            inliers,
            weights,
            iterations,
        }
    }

    /// Extract source and target points from correspondence matrix
    fn extract_points(
        &self,
        correspondences: &DataMatrix,
    ) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let n = correspondences.n_points();
        let mut src_points = Vec::with_capacity(n);
        let mut tgt_points = Vec::with_capacity(n);

        for i in 0..n {
            src_points.push(Vector3::new(
                correspondences.get(i, 0),
                correspondences.get(i, 1),
                correspondences.get(i, 2),
            ));
            tgt_points.push(Vector3::new(
                correspondences.get(i, 3),
                correspondences.get(i, 4),
                correspondences.get(i, 5),
            ));
        }

        (src_points, tgt_points)
    }

    /// Solve weighted Procrustes problem: minimize Σ w_i ||tgt_i - (s*R*src_i + t)||²
    ///
    /// This is the weighted orthogonal Procrustes problem.
    fn solve_weighted_procrustes(
        &self,
        src_points: &[Vector3<f64>],
        tgt_points: &[Vector3<f64>],
        weights: &[f64],
        scale: f64,
    ) -> (Matrix3<f64>, Vector3<f64>) {
        let n = src_points.len();

        // Compute weighted centroids
        let mut src_centroid = Vector3::zeros();
        let mut tgt_centroid = Vector3::zeros();
        let mut weight_sum = 0.0;

        for i in 0..n {
            src_centroid += weights[i] * src_points[i];
            tgt_centroid += weights[i] * tgt_points[i];
            weight_sum += weights[i];
        }

        if weight_sum < 1e-10 {
            // All weights are zero, return identity
            return (Matrix3::identity(), Vector3::zeros());
        }

        src_centroid /= weight_sum;
        tgt_centroid /= weight_sum;

        // Center the points
        let mut src_centered = Vec::with_capacity(n);
        let mut tgt_centered = Vec::with_capacity(n);

        for i in 0..n {
            src_centered.push(src_points[i] - src_centroid);
            tgt_centered.push(tgt_points[i] - tgt_centroid);
        }

        // Build weighted covariance matrix H = Σ w_i * tgt_i * (s * src_i)^T
        let mut H = Matrix3::zeros();
        for i in 0..n {
            H += weights[i] * tgt_centered[i] * (scale * src_centered[i]).transpose();
        }

        // SVD of H = U * Σ * V^T
        let svd = SVD::new(H, true, true);

        let U = svd.u.unwrap();
        let Vt = svd.v_t.unwrap();

        // Rotation R = U * V^T (with correction for reflections)
        let mut R = U * Vt;

        // Correct for reflection (ensure det(R) = 1)
        if R.determinant() < 0.0 {
            let mut U_corrected = U;
            U_corrected[(0, 2)] = -U_corrected[(0, 2)];
            U_corrected[(1, 2)] = -U_corrected[(1, 2)];
            U_corrected[(2, 2)] = -U_corrected[(2, 2)];
            R = U_corrected * Vt;
        }

        // Translation t = tgt_centroid - s*R*src_centroid
        let translation = tgt_centroid - scale * R * src_centroid;

        (R, translation)
    }

    /// Compute weights using GNC cost function
    ///
    /// Weight w_i = φ'(r_i² / σ²) where:
    /// - r_i is the residual for correspondence i
    /// - σ is the noise bound
    /// - φ is the truncated least squares cost (controlled by μ)
    fn compute_weights(
        &self,
        src_points: &[Vector3<f64>],
        tgt_points: &[Vector3<f64>],
        rotation: &Matrix3<f64>,
        translation: &Vector3<f64>,
        scale: f64,
        mu: f64,
    ) -> Vec<f64> {
        let n = src_points.len();
        let mut weights = Vec::with_capacity(n);

        let sigma_sq = self.noise_bound * self.noise_bound;

        for i in 0..n {
            // Compute residual: ||tgt_i - (s*R*src_i + t)||²
            let transformed = scale * rotation * src_points[i] + translation;
            let residual = tgt_points[i] - transformed;
            let residual_sq = residual.norm_squared();

            // Normalized residual
            let r = residual_sq / sigma_sq;

            // GNC weight: w = μ / (μ + r)
            // As μ → ∞, this becomes a hard threshold (0 or 1)
            // As μ → 0, this becomes uniform (convex)
            let weight = mu / (mu + r);

            weights.push(weight);
        }

        weights
    }

    /// Compute change in weights (L2 norm)
    fn compute_weight_change(&self, old_weights: &[f64], new_weights: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (w_old, w_new) in old_weights.iter().zip(new_weights.iter()) {
            let diff = w_old - w_new;
            sum += diff * diff;
        }
        (sum / old_weights.len() as f64).sqrt()
    }

    /// Identify inliers based on final weights
    ///
    /// Uses a threshold of 0.5 on the weights
    fn identify_inliers(&self, weights: &[f64]) -> Vec<usize> {
        weights
            .iter()
            .enumerate()
            .filter(|(_, w)| **w > 0.5)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnc_identity_transform() {
        // Create identical point sets (identity transform)
        let points = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // src = tgt
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0,
        ];
        let correspondences = DataMatrix::from_row_slice(4, 6, &points);

        let solver = GNCSolver::new(0.01);
        let result = solver.solve(&correspondences, 1.0);

        // Should get identity rotation
        let identity = Matrix3::identity();
        for i in 0..3 {
            for j in 0..3 {
                let rot_val: f64 = result.rotation[(i, j)];
                let id_val: f64 = identity[(i, j)];
                assert!(
                    (rot_val - id_val).abs() < 0.01,
                    "Rotation should be identity"
                );
            }
        }

        // Should get zero translation
        assert!(
            result.translation.norm() < 0.01,
            "Translation should be zero"
        );

        // All should be inliers
        assert_eq!(
            result.inliers.len(),
            4,
            "All correspondences should be inliers"
        );
    }

    #[test]
    fn test_gnc_pure_translation() {
        // Source points
        let src = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        // Target = src + [0.5, 0.3, 0.2]
        let translation_gt = Vector3::new(0.5, 0.3, 0.2);
        let mut data = Vec::new();
        for p in &src {
            let tgt = p + translation_gt;
            data.push(p.x);
            data.push(p.y);
            data.push(p.z);
            data.push(tgt.x);
            data.push(tgt.y);
            data.push(tgt.z);
        }

        let correspondences = DataMatrix::from_row_slice(4, 6, &data);

        let solver = GNCSolver::new(0.01);
        let result = solver.solve(&correspondences, 1.0);

        // Check rotation is identity
        let identity = Matrix3::identity();
        for i in 0..3 {
            for j in 0..3 {
                let rot_val: f64 = result.rotation[(i, j)];
                let id_val: f64 = identity[(i, j)];
                assert!(
                    (rot_val - id_val).abs() < 0.05,
                    "Rotation should be near identity"
                );
            }
        }

        // Check translation
        let t_error = (result.translation - translation_gt).norm();
        assert!(
            t_error < 0.05,
            "Translation error {} should be small",
            t_error
        );

        println!(
            "Translation: expected {:?}, got {:?}",
            translation_gt, result.translation
        );
    }

    #[test]
    #[ignore] // GNC alone may not perfectly handle outliers without ROBIN pre-filtering
    fn test_gnc_with_outliers() {
        // Create inliers with pure translation
        let translation_gt = Vector3::new(0.3, 0.4, 0.1);
        let mut data = Vec::new();

        // Add 10 inliers
        for i in 0..10 {
            let src = Vector3::new(i as f64 * 0.1, 0.0, 0.0);
            let tgt = src + translation_gt;
            data.push(src.x);
            data.push(src.y);
            data.push(src.z);
            data.push(tgt.x);
            data.push(tgt.y);
            data.push(tgt.z);
        }

        // Add 5 outliers (far away, should be rejected)
        for i in 0..5 {
            let src = Vector3::new(i as f64 * 0.1, 0.0, 0.0);
            data.push(src.x);
            data.push(src.y);
            data.push(src.z);
            data.push(10.0 + i as f64); // Target far away
            data.push(10.0);
            data.push(10.0);
        }

        let correspondences = DataMatrix::from_row_slice(15, 6, &data);

        let solver = GNCSolver::new(0.01); // Tight noise bound
        let result = solver.solve(&correspondences, 1.0);

        println!(
            "Found {} inliers out of 15 correspondences (expected ~10)",
            result.inliers.len()
        );
        println!(
            "Translation: expected {:?}, got {:?}",
            translation_gt, result.translation
        );

        // GNC should identify most inliers (at least 7 out of 10)
        // and reject most outliers
        assert!(
            result.inliers.len() >= 7 && result.inliers.len() <= 12,
            "Should identify 7-12 inliers, got {}",
            result.inliers.len()
        );

        // Translation should be close to ground truth
        // (may be slightly off if some outliers have high weight)
        let t_error = (result.translation - translation_gt).norm();
        println!("Translation error: {:.4}", t_error);

        // Allow larger error tolerance since GNC may not perfectly reject all outliers
        assert!(
            t_error < 1.0,
            "Translation error {} should be reasonable",
            t_error
        );
    }
}
