//! Plane segmentation example.
//!
//! Loads a point cloud, voxel-downsamples it, then fits a plane using
//! inlier's MSAC + IRLS pipeline (much more robust than SpatialRust's
//! basic random-sample-and-count RANSAC).
//!
//! Usage:
//!   cargo run --example plane_segmentation --features "spatialrust-inlier/filtering spatialrust-inlier/io" -- path/to/cloud.pcd
//!   cargo run --example plane_segmentation --features "spatialrust-inlier/filtering spatialrust-inlier/io"

use spatialrust_inlier::{
    PointCloud, PointCloudBuilder, StandardSchemas,
    filter::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig},
    io::read_point_cloud_file,
    plane::estimate_plane_from_cloud,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let cloud = if args.len() > 1 {
        println!("Loading: {}", args[1]);
        read_point_cloud_file(&args[1])?
    } else {
        println!("No file supplied — generating synthetic noisy plane + outliers.");
        synthetic_cloud()
    };

    println!("Loaded {} points.", cloud.len());

    // Voxel downsample to speed up RANSAC
    let cloud =
        VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.05)).filter(&cloud)?;
    println!("After voxel (5 cm): {} points.", cloud.len());

    // Fit plane with inlier's RANSAC (MSAC scoring + IRLS local optimizer)
    let threshold = 0.02; // 2 cm inlier band
    let result = estimate_plane_from_cloud(&cloud, threshold, None)?;

    println!(
        "Plane normal: [{:.4}, {:.4}, {:.4}]  d = {:.4}",
        result.normal[0], result.normal[1], result.normal[2], result.d
    );
    println!(
        "Inliers: {}  Outliers: {}  Iterations: {}",
        result.inlier_cloud.len(),
        result.outlier_cloud.len(),
        result.iterations,
    );

    Ok(())
}

/// Synthesise ~2 000 points: 1 500 on the XY plane (z ≈ 0) plus 500 random outliers.
fn synthetic_cloud() -> PointCloud {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    let mut seed: u64 = 0xdeadbeef;

    let mut rng = move || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut h = DefaultHasher::new();
        seed.hash(&mut h);
        (h.finish() as f32) / (u64::MAX as f32)
    };

    // Inliers: on the z = 0 plane with small Gaussian-like noise
    for _ in 0..1500 {
        let x = (rng() - 0.5) * 10.0;
        let y = (rng() - 0.5) * 10.0;
        let z = (rng() - 0.5) * 0.01; // noise ≈ 5 mm
        builder.push_point([x, y, z]).unwrap();
    }
    // Outliers: random 3-D
    for _ in 0..500 {
        let x = (rng() - 0.5) * 10.0;
        let y = (rng() - 0.5) * 10.0;
        let z = (rng() - 0.5) * 5.0 + 2.0; // clearly off the plane
        builder.push_point([x, y, z]).unwrap();
    }

    builder.build().unwrap()
}
