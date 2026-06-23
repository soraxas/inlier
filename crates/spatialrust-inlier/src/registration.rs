use inlier::pcr::{PCRConfig, register_rigid};
use spatialrust_core::{PointCloud, SpatialError, SpatialResult};
use spatialrust_registration::{PointCloudRegistration, RegistrationResult};

use crate::convert::{nalgebra_to_isometry3, point_cloud_to_data_matrix};

/// Wraps inlier's full RANSAC registration pipeline as a SpatialRust
/// `PointCloudRegistration` backend.
///
/// Uses MAGSAC scoring + PROSAC sampling + IRLS local optimization —
/// significantly more robust than SpatialRust's built-in basic RANSAC.
pub struct InlierRegistration {
    config: PCRConfig,
}

impl InlierRegistration {
    pub fn new(config: PCRConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self { config: PCRConfig::default() }
    }
}

impl PointCloudRegistration for InlierRegistration {
    fn name(&self) -> &'static str {
        "inlier-ransac"
    }

    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult> {
        let src = point_cloud_to_data_matrix(source)?;
        let dst = point_cloud_to_data_matrix(target)?;

        let result = register_rigid(&src, &dst, &self.config)
            .ok_or_else(|| SpatialError::InvalidArgument("registration failed".into()))?;

        let transform = nalgebra_to_isometry3(&result.rotation, &result.translation);
        let fitness = result.inlier_count as f64
            / result.total_correspondences.max(1) as f64;

        Ok(RegistrationResult { transform, fitness, iterations: 0, converged: true })
    }
}
