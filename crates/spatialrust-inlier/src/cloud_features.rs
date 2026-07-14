//! Feature estimation — normals, ISS keypoints, and boundary detection.
//!
//! Normal estimation is the most common preprocessing step before inlier's
//! NAPSAC or Progressive NAPSAC samplers, which use spatial neighbourhood
//! structure to draw geometrically coherent samples.
//!
//! # Example
//!
//! ```rust,ignore
//! use spatialrust_inlier::cloud_features::{NormalEstimator, NormalEstimationConfig};
//! use spatialrust_inlier::convert::point_cloud_to_data_matrix;
//!
//! // Estimate normals with 16-NN, then run inlier's RANSAC on the augmented cloud
//! let (cloud_with_normals, _diag) =
//!     NormalEstimator::new(NormalEstimationConfig::k_neighbors(16))
//!         .estimate_with_diagnostics(&cloud)?;
//! let data = point_cloud_to_data_matrix(&cloud_with_normals)?;
//! ```

pub use spatialrust_features::{
    BoundaryConfig, BoundaryDetector, BoundaryResult, FeatureEstimator, IssKeypointConfig,
    IssKeypointDetector, IssKeypointResult, KdTreeNeighborhood, NeighborhoodProvider,
    NormalEstimationConfig, NormalEstimationResult, NormalEstimator, NormalOrientationConfig,
    orient_normal_towards_viewpoint, orient_normals_consistent,
};

#[cfg(feature = "cloud-features-gpu")]
pub use spatialrust_features::GpuNormalEstimator;
