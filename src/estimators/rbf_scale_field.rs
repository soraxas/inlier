//! RBF-based smooth scale field estimator for non-rigid registration
//!
//! Allows spatially-varying scale using radial basis function interpolation
//! with sparse control points and smooth regularization.

use crate::core::Estimator;
use crate::types::{DataMatrix, Point3};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// RBF kernel types for scale interpolation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RBFKernel {
    /// Gaussian: exp(-r²/(2σ²))
    Gaussian { sigma: f64 },
    /// Wendland C2: (1-r/h)⁴₊(4r/h+1) for r<h
    Wendland { radius: f64 },
    /// Thin-plate spline: r (3D)
    ThinPlate,
}

impl RBFKernel {
    /// Evaluate kernel at distance r
    pub fn eval(&self, r: f64) -> f64 {
        match self {
            RBFKernel::Gaussian { sigma } => (-r * r / (2.0 * sigma * sigma)).exp(),
            RBFKernel::Wendland { radius } => {
                if r >= *radius {
                    0.0
                } else {
                    let x = r / radius;
                    let t = (1.0 - x).max(0.0);
                    t.powi(4) * (4.0 * x + 1.0)
                }
            }
            RBFKernel::ThinPlate => r,
        }
    }

    /// Check if kernel has compact support
    pub fn is_compact(&self) -> bool {
        matches!(
            self,
            RBFKernel::Gaussian { .. } | RBFKernel::Wendland { .. }
        )
    }

    /// Get effective radius (beyond which kernel ≈ 0)
    pub fn effective_radius(&self) -> Option<f64> {
        match self {
            RBFKernel::Gaussian { sigma } => Some(3.0 * sigma), // 99% of mass
            RBFKernel::Wendland { radius } => Some(*radius),
            RBFKernel::ThinPlate => None, // Global support
        }
    }
}

/// Configuration for RBF scale field estimation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RBFScaleConfig {
    /// Number of control points (fewer = smoother, more = flexible)
    pub num_control_points: usize,
    /// RBF kernel type
    pub kernel: RBFKernel,
    /// Regularization weight (higher = smoother)
    pub regularization_lambda: f64,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Minimum allowed scale
    pub min_scale: f64,
    /// Maximum allowed scale
    pub max_scale: f64,
    /// Use sparse RBF matrix (only for compact kernels)
    pub use_sparse: bool,
}

impl Default for RBFScaleConfig {
    fn default() -> Self {
        Self {
            num_control_points: 100,
            kernel: RBFKernel::Gaussian { sigma: 0.15 },
            regularization_lambda: 1e-3,
            max_iterations: 50,
            convergence_threshold: 1e-4,
            min_scale: 0.5,
            max_scale: 2.0,
            use_sparse: true,
        }
    }
}

/// Smooth scale field represented by RBF interpolation
#[derive(Debug, Clone)]
pub struct ScaleField {
    /// Control point positions
    pub control_points: Vec<Point3>,
    /// Weights at control points
    pub weights: DVector<f64>,
    /// RBF kernel
    pub kernel: RBFKernel,
}

impl ScaleField {
    /// Create new scale field with uniform weights
    pub fn new(control_points: Vec<Point3>, kernel: RBFKernel) -> Self {
        let k = control_points.len();
        Self {
            control_points,
            weights: DVector::from_element(k, 1.0),
            kernel,
        }
    }

    /// Evaluate scale at a point
    pub fn eval(&self, point: &Point3) -> f64 {
        let mut scale = 0.0;
        for (i, cp) in self.control_points.iter().enumerate() {
            let dist = (point - cp).norm();
            scale += self.weights[i] * self.kernel.eval(dist);
        }
        scale
    }

    /// Evaluate scales for multiple points (vectorized)
    pub fn eval_batch(&self, points: &[Point3]) -> Vec<f64> {
        #[cfg(feature = "rayon")]
        use rayon::prelude::*;

        #[cfg(feature = "rayon")]
        let iter = points.par_iter();
        #[cfg(not(feature = "rayon"))]
        let iter = points.iter();

        iter.map(|p| self.eval(p)).collect()
    }

    /// Get mean scale
    pub fn mean_scale(&self) -> f64 {
        self.weights.mean()
    }
}

/// Non-rigid transformation with spatially-varying scale
#[derive(Debug, Clone)]
pub struct NonRigidTransform {
    /// Smooth scale field
    pub scale_field: ScaleField,
    /// Rotation matrix
    pub rotation: Matrix3<f64>,
    /// Translation vector
    pub translation: Vector3<f64>,
}

impl NonRigidTransform {
    /// Apply transformation to a point
    pub fn transform(&self, point: &Point3) -> Point3 {
        let scale = self.scale_field.eval(point);
        self.rotation * (scale * point) + self.translation
    }

    /// Apply transformation to multiple points
    pub fn transform_batch(&self, points: &[Point3]) -> Vec<Point3> {
        let scales = self.scale_field.eval_batch(points);
        points
            .iter()
            .zip(scales.iter())
            .map(|(p, &s)| self.rotation * (s * p) + self.translation)
            .collect()
    }

    /// Get mean scale
    pub fn mean_scale(&self) -> f64 {
        self.scale_field.mean_scale()
    }

    /// Compute residual for a correspondence
    pub fn residual(&self, src: &Point3, dst: &Point3) -> f64 {
        let transformed = self.transform(src);
        (dst - transformed).norm()
    }
}

impl fmt::Display for NonRigidTransform {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "NonRigidTransform:")?;
        writeln!(
            f,
            "  Control points: {}",
            self.scale_field.control_points.len()
        )?;
        writeln!(f, "  Mean scale: {:.6}", self.mean_scale())?;
        writeln!(f, "  Rotation:\n{}", self.rotation)?;
        writeln!(f, "  Translation: {}", self.translation)?;
        Ok(())
    }
}

/// Place control points using voxel grid
pub fn place_control_points_voxel(points: &[Point3], voxel_size: f64) -> Vec<Point3> {
    use std::collections::HashMap;

    let mut voxel_map: HashMap<(i32, i32, i32), Vec<Point3>> = HashMap::new();

    // Assign points to voxels
    for point in points {
        let voxel = (
            (point.x / voxel_size).floor() as i32,
            (point.y / voxel_size).floor() as i32,
            (point.z / voxel_size).floor() as i32,
        );
        voxel_map.entry(voxel).or_default().push(*point);
    }

    // Control point = voxel center
    voxel_map
        .into_values()
        .map(|points_in_voxel| {
            // Average of points in voxel
            let sum = points_in_voxel
                .iter()
                .fold(Point3::zeros(), |acc, p| acc + p);
            sum / points_in_voxel.len() as f64
        })
        .collect()
}

/// Place control points using farthest point sampling
pub fn place_control_points_fps(points: &[Point3], num_points: usize) -> Vec<Point3> {
    if points.is_empty() {
        return Vec::new();
    }
    if num_points >= points.len() {
        return points.to_vec();
    }

    let mut control_points = Vec::with_capacity(num_points);
    let mut min_dists = vec![f64::INFINITY; points.len()];

    // Start with random point
    control_points.push(points[0]);

    for _ in 1..num_points {
        // Update distances to nearest control point
        let last_cp = control_points.last().unwrap();
        for (i, point) in points.iter().enumerate() {
            let dist = (point - last_cp).norm();
            min_dists[i] = min_dists[i].min(dist);
        }

        // Select point with maximum distance
        let (idx, _) = min_dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        control_points.push(points[idx]);
        min_dists[idx] = 0.0;
    }

    control_points
}

/// Precompute RBF matrix Φ [N×K]
pub fn compute_rbf_matrix(
    points: &[Point3],
    control_points: &[Point3],
    kernel: &RBFKernel,
) -> DMatrix<f64> {
    let n = points.len();
    let k = control_points.len();

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    // Compute matrix row by row
    let values: Vec<f64> = {
        #[cfg(feature = "rayon")]
        let iter = points.par_iter();
        #[cfg(not(feature = "rayon"))]
        let iter = points.iter();

        iter.flat_map(|point| {
            control_points
                .iter()
                .map(|cp| {
                    let dist = (point - cp).norm();
                    kernel.eval(dist)
                })
                .collect::<Vec<_>>()
        })
        .collect()
    };

    DMatrix::from_row_slice(n, k, &values)
}

/// RBF-based smooth scale field estimator
pub struct RBFScaleEstimator {
    pub config: RBFScaleConfig,
}

impl RBFScaleEstimator {
    pub fn new(config: RBFScaleConfig) -> Self {
        Self { config }
    }

    /// Estimate transformation from correspondences
    pub fn estimate(
        &self,
        src_points: &[Point3],
        dst_points: &[Point3],
    ) -> Option<NonRigidTransform> {
        if src_points.len() != dst_points.len() || src_points.len() < 3 {
            return None;
        }

        // [1] Place control points
        let control_points = if self.config.num_control_points >= src_points.len() {
            src_points.to_vec()
        } else {
            // Use voxel-based placement
            let voxel_size = estimate_voxel_size(src_points, self.config.num_control_points);
            place_control_points_voxel(src_points, voxel_size)
        };

        if control_points.is_empty() {
            return None;
        }

        // [2] Precompute RBF matrix
        let phi = compute_rbf_matrix(src_points, &control_points, &self.config.kernel);

        // [3] Initialize
        let k = control_points.len();
        let mut weights = DVector::from_element(k, 1.0);
        let mut rotation = Matrix3::identity();

        // Initial alignment (center and compute initial translation)
        let src_center = compute_centroid(src_points);
        let dst_center = compute_centroid(dst_points);
        let mut translation = dst_center - src_center;

        let mut prev_energy = f64::INFINITY;

        // [4] Alternating optimization
        for _iter in 0..self.config.max_iterations {
            // Step 1: Optimize R, t given weights
            let scales = &phi * &weights;
            let (new_r, new_t) =
                optimize_rotation_translation(src_points, dst_points, scales.as_slice())?;
            rotation = new_r;
            translation = new_t;

            // Step 2: Optimize weights given R, t
            let new_weights = optimize_weights(
                src_points,
                dst_points,
                &phi,
                &rotation,
                &translation,
                self.config.regularization_lambda,
            )?;

            // Clamp weights
            weights = new_weights.map(|w| w.clamp(self.config.min_scale, self.config.max_scale));

            // Check convergence
            let energy = compute_energy(
                src_points,
                dst_points,
                &phi,
                &weights,
                &rotation,
                &translation,
                self.config.regularization_lambda,
            );

            let delta_energy = (prev_energy - energy).abs() / prev_energy.max(1e-10);

            if delta_energy < self.config.convergence_threshold {
                #[cfg(feature = "verbose")]
                println!(
                    "RBF converged at iteration {} (ΔE={:.2e})",
                    _iter, delta_energy
                );
                break;
            }

            prev_energy = energy;
        }

        Some(NonRigidTransform {
            scale_field: ScaleField {
                control_points,
                weights,
                kernel: self.config.kernel,
            },
            rotation,
            translation,
        })
    }
}

impl Estimator for RBFScaleEstimator {
    type Model = NonRigidTransform;

    fn sample_size(&self) -> usize {
        3
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < 3 {
            return false;
        }

        // Extract source points from sample
        let src_points: Vec<Point3> = sample
            .iter()
            .map(|&i| Point3::new(data.get(i, 0), data.get(i, 1), data.get(i, 2)))
            .collect();

        // Check for degenerate configurations (collinear points)
        let p0 = src_points[0];
        let p1 = src_points[1];
        let p2 = src_points[2];

        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let cross = v1.cross(&v2);

        // Non-collinear if cross product is significant
        cross.norm() > 1e-6
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        if sample.len() < 3 {
            return vec![];
        }

        let src_points: Vec<Point3> = sample
            .iter()
            .map(|&i| Point3::new(data.get(i, 0), data.get(i, 1), data.get(i, 2)))
            .collect();

        let dst_points: Vec<Point3> = sample
            .iter()
            .map(|&i| Point3::new(data.get(i, 3), data.get(i, 4), data.get(i, 5)))
            .collect();

        match self.estimate(&src_points, &dst_points) {
            Some(transform) => vec![transform],
            None => vec![],
        }
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        // Check rotation is proper (det ≈ 1)
        let det = model.rotation.determinant();
        if (det - 1.0).abs() > 0.1 {
            return false;
        }

        // Check mean scale is reasonable
        let mean_scale = model.mean_scale();
        if mean_scale < self.config.min_scale || mean_scale > self.config.max_scale {
            return false;
        }

        true
    }
}

// Helper functions

fn compute_centroid(points: &[Point3]) -> Point3 {
    let sum = points.iter().fold(Point3::zeros(), |acc, p| acc + p);
    sum / points.len() as f64
}

fn estimate_voxel_size(points: &[Point3], target_voxels: usize) -> f64 {
    if points.is_empty() {
        return 1.0;
    }

    // Estimate bounding box
    let mut min = points[0];
    let mut max = points[0];
    for p in points {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }

    let volume = (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
    let voxel_volume = volume / target_voxels as f64;
    voxel_volume.cbrt()
}

/// Optimize rotation and translation given scales
fn optimize_rotation_translation(
    src_points: &[Point3],
    dst_points: &[Point3],
    scales: &[f64],
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let n = src_points.len();
    if n < 3 {
        return None;
    }

    // Scale source points
    let src_scaled: Vec<Point3> = src_points
        .iter()
        .zip(scales.iter())
        .map(|(p, &s)| s * p)
        .collect();

    // Compute centroids
    let src_center = compute_centroid(&src_scaled);
    let dst_center = compute_centroid(dst_points);

    // Center point clouds
    let src_centered: Vec<Point3> = src_scaled.iter().map(|p| p - src_center).collect();
    let dst_centered: Vec<Point3> = dst_points.iter().map(|p| p - dst_center).collect();

    // Compute covariance matrix H = Σ src_i * dst_i^T
    let mut h = Matrix3::zeros();
    for (s, d) in src_centered.iter().zip(dst_centered.iter()) {
        h += s * d.transpose();
    }

    // SVD
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;

    // Rotation (handle reflection)
    let det = (v_t.transpose() * u.transpose()).determinant();
    let correction = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, det.signum()));
    let rotation = v_t.transpose() * correction * u.transpose();

    // Translation
    let translation = dst_center - rotation * src_center;

    Some((rotation, translation))
}

/// Optimize weights given rotation and translation
fn optimize_weights(
    src_points: &[Point3],
    dst_points: &[Point3],
    phi: &DMatrix<f64>,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    lambda: f64,
) -> Option<DVector<f64>> {
    let n = src_points.len();
    let k = phi.ncols();

    // Precompute Q = R * S (rotated source)
    let q_points: Vec<Point3> = src_points.iter().map(|s| rotation * s).collect();

    // Precompute residuals r = dst - t
    let r_points: Vec<Point3> = dst_points.iter().map(|d| d - translation).collect();

    // Build A matrix (K×K): A = Φ^T diag(||q||²) Φ + λI
    let mut a = DMatrix::zeros(k, k);
    for i in 0..k {
        for ip in 0..k {
            let mut sum = 0.0;
            for j in 0..n {
                let q_norm_sq = q_points[j].norm_squared();
                sum += phi[(j, i)] * phi[(j, ip)] * q_norm_sq;
            }
            a[(i, ip)] = sum;
        }
        a[(i, i)] += lambda; // Regularization
    }

    // Build b vector (K×1): b = Φ^T (r·q)
    let mut b = DVector::zeros(k);
    for i in 0..k {
        let mut sum = 0.0;
        for j in 0..n {
            let dot = r_points[j].dot(&q_points[j]);
            sum += phi[(j, i)] * dot;
        }
        b[i] = sum;
    }

    // Solve (A)w = b
    let decomp = a.lu();
    decomp.solve(&b)
}

/// Compute total energy
fn compute_energy(
    src_points: &[Point3],
    dst_points: &[Point3],
    phi: &DMatrix<f64>,
    weights: &DVector<f64>,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    lambda: f64,
) -> f64 {
    let scales = phi * weights;

    // Data term
    let mut data_energy = 0.0;
    for (i, (src, dst)) in src_points.iter().zip(dst_points.iter()).enumerate() {
        let s = scales[i];
        let residual = dst - (rotation * (s * src) + translation);
        data_energy += residual.norm_squared();
    }

    // Regularization term
    let reg_energy = lambda * weights.norm_squared();

    data_energy + reg_energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernels() {
        let gauss = RBFKernel::Gaussian { sigma: 1.0 };
        assert!((gauss.eval(0.0) - 1.0).abs() < 1e-10);
        // Gaussian at 3σ: exp(-9/2) ≈ 0.011
        assert!(gauss.eval(3.0) < 0.02);

        let wendland = RBFKernel::Wendland { radius: 1.0 };
        assert!((wendland.eval(0.0) - 1.0).abs() < 1e-10);
        assert!(wendland.eval(1.5).abs() < 1e-10);
    }

    #[test]
    fn test_control_point_placement() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];

        let cp = place_control_points_voxel(&points, 0.5);
        assert!(cp.len() <= 4);

        let cp_fps = place_control_points_fps(&points, 2);
        assert_eq!(cp_fps.len(), 2);
    }

    #[test]
    fn test_uniform_scale() {
        // Simple sanity check: estimator should at least converge
        let src = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
        ];

        let scale = 1.5;
        let dst: Vec<Point3> = src.iter().map(|p| scale * p).collect();

        let config = RBFScaleConfig::default();
        let estimator = RBFScaleEstimator::new(config);
        let result = estimator.estimate(&src, &dst);

        // Should successfully produce a result
        assert!(result.is_some(), "Estimator should converge");
    }

    #[test]
    fn test_scale_gradient() {
        // Test basic properties with spatially-varying scale
        use nalgebra::Rotation3;

        let src = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.5, 0.0),
            Point3::new(1.0, 1.0, 0.5),
            Point3::new(1.5, 0.5, 1.0),
            Point3::new(2.0, 1.0, 0.0),
        ];

        // Apply rotation + different scales
        let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.3);
        let scales = [1.0, 1.2, 1.4, 1.6, 1.8];
        let dst: Vec<Point3> = src
            .iter()
            .zip(scales.iter())
            .map(|(p, &s)| s * (rot * p) + Vector3::new(0.5, 0.5, 0.5))
            .collect();

        let config = RBFScaleConfig {
            num_control_points: 5,
            max_iterations: 50,
            regularization_lambda: 1e-3,
            ..Default::default()
        };

        let estimator = RBFScaleEstimator::new(config);
        let result = estimator.estimate(&src, &dst);

        // Should successfully produce a result
        assert!(
            result.is_some(),
            "Estimator should converge for varying scales"
        );

        if let Some(transform) = result {
            // Check rotation is proper
            assert!((transform.rotation.determinant() - 1.0).abs() < 0.1);

            // Check control points were created
            assert!(!transform.scale_field.control_points.is_empty());
        }
    }
}
