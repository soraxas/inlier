//! Point cloud segmentation — plane/sphere/cylinder RANSAC, clustering, and ground removal.
//!
//! SpatialRust's segmenters are re-exported directly. For plane fitting with
//! inlier's advanced RANSAC (MAGSAC + NAPSAC + IRLS), use [`InlierRegistration`]
//! from the `registration` module combined with a custom plane estimator.
//!
//! # Example
//!
//! ```rust,ignore
//! use spatialrust_inlier::segmentation::{RansacPlaneSegmenter, RansacPlaneConfig};
//!
//! let seg = RansacPlaneSegmenter::new(RansacPlaneConfig::with_distance_threshold(0.02))
//!     .segment(&cloud)?;
//! println!("plane normal {:?}, inliers {}", seg.model.normal, seg.inlier_count);
//!
//! use spatialrust_inlier::segmentation::{EuclideanClusterExtractor, EuclideanClusterConfig};
//! let clusters = EuclideanClusterExtractor::new(
//!     EuclideanClusterConfig::with_tolerance(0.05, 50)
//! ).extract(&seg.outliers)?;
//! println!("{} clusters found", clusters.cluster_count);
//! ```

pub use spatialrust_segmentation::{
    extract_indices, extract_mask, with_labels,
    PlaneModel, RansacPlaneSegmenter, RansacPlaneConfig, RansacPlaneSegmentation,
    EuclideanClusterExtractor, EuclideanClusterConfig, EuclideanClusterResult,
    DbscanSegmenter, DbscanConfig, DbscanResult,
    GroundSegmenter, GroundConfig, GroundSegmentation, UpAxis,
    RegionGrowingSegmenter, RegionGrowingConfig, RegionGrowingResult,
    RansacSphereSegmenter, RansacCylinderSegmenter,
};
