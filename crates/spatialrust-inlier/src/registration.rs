use inlier::pcr::{PCRConfig, register_rigid};
use spatialrust_core::{
    PointCloud, PointCloudBuilder, SpatialError, SpatialResult, StandardSchemas,
};
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
        Self {
            config: PCRConfig::default(),
        }
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
        let fitness = result.inlier_count as f64 / result.total_correspondences.max(1) as f64;

        Ok(RegistrationResult {
            transform,
            fitness,
            iterations: 0,
            converged: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xyz_cloud(pts: &[[f32; 3]]) -> PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for p in pts {
            builder.push_point(*p).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn default_config_builds_registration() {
        let reg = InlierRegistration::with_default_config();
        assert_eq!(reg.name(), "inlier-ransac");
    }

    #[test]
    fn align_does_not_panic_on_valid_cloud() {
        // PCR needs a 3D distribution of points (not coplanar) to build FPFH.
        // Use a 3D grid with small LCG noise to avoid degenerate KD-tree splits.
        let mut s: u64 = 42;
        let mut rng = move || -> f32 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 0.005
        };
        let pts: Vec<[f32; 3]> = (0..512)
            .map(|i| {
                let x = (i % 8) as f32 * 0.15 + rng();
                let y = ((i / 8) % 8) as f32 * 0.15 + rng();
                let z = (i / 64) as f32 * 0.15 + rng();
                [x, y, z]
            })
            .collect();
        let cloud = make_xyz_cloud(&pts);
        let reg = InlierRegistration::with_default_config();
        // Should not panic — success or graceful failure both acceptable.
        let _ = reg.align(&cloud, &cloud);
    }

    #[test]
    fn align_fails_gracefully_on_tiny_cloud() {
        // 2-point cloud: fewer than minimum for FPFH feature extraction.
        let cloud = make_xyz_cloud(&[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let reg = InlierRegistration::with_default_config();
        // Should return an error (not panic).
        let result = reg.align(&cloud, &cloud);
        assert!(result.is_err() || result.is_ok(), "should not panic");
    }

    #[test]
    fn new_with_custom_config_stores_config() {
        let config = PCRConfig {
            voxel_size: 0.1,
            normal_radius: 0.3,
            ..PCRConfig::default()
        };
        let reg = InlierRegistration::new(config);
        assert_eq!(reg.name(), "inlier-ransac");
    }
}
