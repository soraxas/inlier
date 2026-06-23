//! Occupancy grids and LiDAR range images for ML preprocessing.
//!
//! # Example
//!
//! ```rust,ignore
//! use spatialrust_inlier::voxelize::{voxelize, VoxelGridConfig};
//!
//! // 3-D occupancy grid at 6 cm resolution
//! let grid = voxelize(&cloud, VoxelGridConfig::new(0.06))?;
//! println!("dims {:?}, occupied {}", grid.dims, grid.occupied_count());
//!
//! // LiDAR spherical range image (HDL-64 style)
//! use spatialrust_inlier::voxelize::{range_image, RangeImageConfig};
//! let img = range_image(&cloud, RangeImageConfig::new(1024, 64))?;
//! ```

pub use spatialrust_voxelize::{
    OccupancyGrid, VoxelFill, VoxelGridConfig, voxelize,
    RangeImage, RangeImageConfig, range_image,
};
