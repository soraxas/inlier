//! Bridge between inlier's RANSAC engine and SpatialRust's point cloud pipeline.
//!
//! # Feature flags
//!
//! | Flag | Enables |
//! |------|---------|
//! | `io` | PCD / PLY / LAS / LAZ file reading and writing |
//! | `io-e57` | E57 support (implies `io`) |
//! | `io-copc` | Cloud Optimized Point Cloud support (implies `io`) |
//! | `filtering` | Voxel downsampling, outlier removal, crop, FPS |
//! | `filtering-gpu` | GPU-accelerated voxel filter (implies `filtering`) |
//! | `voxelize` | Occupancy grids and LiDAR range images |
//! | `cloud-features` | Normal estimation, ISS keypoints, boundary detection |
//! | `cloud-features-gpu` | GPU normal estimation (implies `cloud-features`) |
//! | `segmentation` | Plane/sphere/cylinder RANSAC, DBSCAN, Euclidean clustering, ground segmentation |
//! | `registration` | `InlierRegistration` — inlier's full RANSAC as a SpatialRust backend |
//! | `full` | All of the above (except GPU variants) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use spatialrust_inlier::{
//!     io::read_point_cloud_file,
//!     filter::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig},
//!     convert::point_cloud_to_data_matrix,
//! };
//!
//! let cloud = read_point_cloud_file("scan.las").unwrap();
//! let downsampled = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.05))
//!     .filter(&cloud).unwrap();
//! let data = point_cloud_to_data_matrix(&downsampled).unwrap();
//! // feed `data` into inlier's RANSAC estimators
//! ```

pub mod convert;
pub mod plane;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "filtering")]
pub mod filter;

#[cfg(feature = "voxelize")]
pub mod voxelize;

#[cfg(feature = "cloud-features")]
pub mod cloud_features;

#[cfg(feature = "segmentation")]
pub mod segmentation;

#[cfg(feature = "registration")]
pub mod registration;

// Flat re-exports for the most common conversion functions.
pub use convert::{
    data_matrix_to_point_cloud, nalgebra_to_isometry3, point_cloud_to_data_matrix,
    point_cloud_with_normals_to_data_matrix,
};
pub use plane::{PlaneResult, estimate_plane_from_cloud, fit_plane_msac, fit_plane_magsac_raw};
pub use inlier::MetasacSettings;
pub use spatialrust_core::{
    FieldSemantic, PointBuffer, PointCloud, PointCloudBuilder, SpatialError, SpatialResult, StandardSchemas,
};

#[cfg(feature = "registration")]
pub use registration::InlierRegistration;
