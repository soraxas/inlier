//! High-level Point Cloud Registration (PCR) API
//!
//! Provides easy-to-use functions for point cloud registration:
//! - Rigid registration (KISS-Matcher)
//! - Non-rigid registration with scale variation (SIPFH + RBF)
//!
//! # Example
//! ```ignore
//! use inlier::pcr::{register_rigid, register_nonrigid, PCRConfig};
//!
//! let result = register_rigid(&src_points, &dst_points, &PCRConfig::default())?;
//! println!("Rotation: {}", result.rotation);
//! println!("Translation: {:?}", result.translation);
//! println!("Inliers: {}", result.inlier_count);
//! ```

use crate::estimators::rbf_scale_field::{NonRigidTransform, RBFKernel, RBFScaleConfig};
use crate::matcher::config::KISSMatcherConfig;
use crate::matcher::kiss_matcher_full_pipeline;
use crate::matcher::pipeline_nonrigid::{
    FeatureMethod, NonRigidKISSConfig, nonrigid_kiss_matcher_pipeline,
};
use crate::matcher::sipfh::SIPFHConfig;
use crate::types::DataMatrix;
use nalgebra::{Matrix3, Vector3};

/// High-level PCR configuration
#[derive(Clone, Debug)]
pub struct PCRConfig {
    /// Voxel size for downsampling (set to 0.0 to disable)
    pub voxel_size: f64,
    /// Normal estimation radius
    pub normal_radius: f64,
    /// Feature descriptor radius
    pub feature_radius: f64,
    /// Feature matching ratio test threshold (0.0-1.0, lower = stricter)
    pub ratio_threshold: f64,
    /// ROBIN outlier rejection noise bound
    pub noise_bound: f64,
    /// Use scale-invariant features (SIPFH) - recommended for non-rigid
    pub use_scale_invariant_features: bool,
}

impl Default for PCRConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            normal_radius: 0.15,
            feature_radius: 0.3,
            ratio_threshold: 0.9,
            noise_bound: 0.05,
            use_scale_invariant_features: false,
        }
    }
}

/// Rigid registration result
#[derive(Clone, Debug)]
pub struct RigidRegistrationResult {
    /// 3×3 rotation matrix
    pub rotation: Matrix3<f64>,
    /// Translation vector
    pub translation: Vector3<f64>,
    /// Uniform scale factor
    pub scale: f64,
    /// Number of inlier correspondences
    pub inlier_count: usize,
    /// Total number of initial correspondences
    pub total_correspondences: usize,
}

/// Non-rigid registration result
#[derive(Clone, Debug)]
pub struct NonRigidRegistrationResult {
    /// Spatially-varying transformation (RBF scale field + rigid)
    pub transform: NonRigidTransform,
    /// Mean scale across all points
    pub mean_scale: f64,
    /// Standard deviation of scale field
    pub scale_std: f64,
    /// Number of inlier correspondences
    pub inlier_count: usize,
    /// Total number of initial correspondences
    pub total_correspondences: usize,
}

/// Register point clouds with rigid transformation (rotation + translation + uniform scale)
///
/// # Arguments
/// * `src` - Source point cloud (N × 3)
/// * `dst` - Target point cloud (M × 3)
/// * `config` - Registration configuration
///
/// # Returns
/// Registration result with transformation and inlier count, or None if failed
pub fn register_rigid(
    src: &DataMatrix,
    dst: &DataMatrix,
    config: &PCRConfig,
) -> Option<RigidRegistrationResult> {
    let kiss_config = KISSMatcherConfig {
        voxel_size: config.voxel_size,
        normal_radius: config.normal_radius,
        fpfh_radius: config.feature_radius,
        ratio_threshold: config.ratio_threshold,
        robin_noise_bound: config.noise_bound,
        solver_noise_bound: config.noise_bound * 0.5,
        the_linearity: 10.0,
        ..Default::default()
    };

    let result = kiss_matcher_full_pipeline(src, dst, &kiss_config)?;

    Some(RigidRegistrationResult {
        rotation: result.rotation,
        translation: result.translation,
        scale: result.scale,
        inlier_count: result.inlier_indices.len(),
        total_correspondences: result.n_correspondences_initial,
    })
}

/// Register point clouds with non-rigid transformation (spatially-varying scale + rigid)
///
/// Suitable for:
/// - Thermal expansion/contraction
/// - Biological growth
/// - Non-uniform deformation
///
/// # Arguments
/// * `src` - Source point cloud (N × 3)
/// * `dst` - Target point cloud (M × 3)
/// * `config` - Registration configuration
///
/// # Returns
/// Registration result with RBF scale field and inlier count, or None if failed
pub fn register_nonrigid(
    src: &DataMatrix,
    dst: &DataMatrix,
    config: &PCRConfig,
) -> Option<NonRigidRegistrationResult> {
    let feature_method = if config.use_scale_invariant_features {
        FeatureMethod::SIPFH(SIPFHConfig {
            num_octaves: 3,
            scales_per_octave: 3,
            initial_sigma: 0.05,
            dog_threshold: 0.015,
            edge_threshold: 10.0,
            fpfh_radius: config.feature_radius,
            the_linearity: 0.9,
            fpfh_bins: 11,
            scale_weight: 0.5,
        })
    } else {
        FeatureMethod::FasterPFH
    };

    let nonrigid_config = NonRigidKISSConfig {
        base: KISSMatcherConfig {
            voxel_size: config.voxel_size,
            normal_radius: config.normal_radius,
            fpfh_radius: config.feature_radius,
            ratio_threshold: config.ratio_threshold,
            robin_noise_bound: config.noise_bound,
            solver_noise_bound: config.noise_bound * 0.5,
            the_linearity: 10.0,
            ..Default::default()
        },
        rbf: RBFScaleConfig {
            num_control_points: 10,
            kernel: RBFKernel::Gaussian { sigma: 0.2 },
            regularization_lambda: 1e-3,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            min_scale: 0.8,
            max_scale: 1.5,
            use_sparse: true,
        },
        feature_method,
    };

    let result = nonrigid_kiss_matcher_pipeline(src, dst, &nonrigid_config)?;

    Some(NonRigidRegistrationResult {
        transform: result.transform,
        mean_scale: result.mean_scale,
        scale_std: result.scale_std,
        inlier_count: result.n_correspondences_final,
        total_correspondences: result.n_correspondences_initial,
    })
}

/// Quick rigid registration with default parameters
pub fn register_rigid_auto(src: &DataMatrix, dst: &DataMatrix) -> Option<RigidRegistrationResult> {
    register_rigid(src, dst, &PCRConfig::default())
}

/// Quick non-rigid registration with default parameters and scale-invariant features
pub fn register_nonrigid_auto(
    src: &DataMatrix,
    dst: &DataMatrix,
) -> Option<NonRigidRegistrationResult> {
    let mut config = PCRConfig::default();
    config.use_scale_invariant_features = true;
    register_nonrigid(src, dst, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcr_config_default() {
        let config = PCRConfig::default();
        assert_eq!(config.voxel_size, 0.05);
        assert!(!config.use_scale_invariant_features);
    }
}
