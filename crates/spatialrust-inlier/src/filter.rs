//! Point cloud filtering — voxel downsampling, outlier removal, crop, and FPS.
//!
//! All filters implement [`PointCloudFilter`] and return a new [`PointCloud`].
//!
//! # Example
//!
//! ```rust,ignore
//! use spatialrust_inlier::filter::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig};
//!
//! // CPU centroid voxel grid at 5 cm leaf size
//! let downsampled = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.05))
//!     .filter(&cloud)?;
//! ```

pub use spatialrust_filtering::PointCloudFilter;

pub use spatialrust_filtering::{
    Aabb, CropBox, PassThrough,
    FarthestPointSampling, FarthestPointSamplingConfig,
    StatisticalOutlierRemoval, StatisticalOutlierConfig,
    RadiusOutlierRemoval, RadiusOutlierConfig,
    VoxelGridDownsample, VoxelGridDownsampleConfig, VoxelAggregationMode, AttributeAggregation,
};
