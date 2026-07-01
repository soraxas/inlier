//! File IO — read and write point clouds in PCD, PLY, LAS/LAZ, E57, and COPC formats.
//!
//! The top-level helpers auto-detect the format from the file extension.

pub use spatialrust_io::{
    IoError, PointCloudFileFormat, ReadOptions, WriteOptions,
    detect_point_cloud_format, read_point_cloud_file, read_point_cloud_file_with_format,
    write_point_cloud_file, write_point_cloud_file_with_format,
};

#[cfg(feature = "io")]
pub use spatialrust_io::{
    read_pcd_file, write_pcd_file,
    read_ply, read_ply_file, write_ply_file,
    read_las_file, write_las_file,
};

#[cfg(feature = "io-e57")]
pub use spatialrust_io::{read_e57_file, write_e57_file};

#[cfg(feature = "io-copc")]
pub use spatialrust_io::{
    CopcBounds, CopcQuery,
    read_copc_file, read_copc_file_in_bounds, read_copc_file_with_query,
    write_copc_file,
};
