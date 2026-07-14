//! Full spatial pipeline example.
//!
//! Demonstrates the end-to-end workflow:
//!   Load → Voxel downsample → Estimate normals → Plane segmentation
//!   → Euclidean clustering on residuals → Occupancy grid of each cluster
//!
//! Usage:
//!   cargo run --example full_pipeline --features "spatialrust-inlier/full" -- scan.las
//!   cargo run --example full_pipeline --features "spatialrust-inlier/full"   # synthetic data

use spatialrust_inlier::{
    PointCloud, PointCloudBuilder, StandardSchemas,
    cloud_features::{NormalEstimationConfig, NormalEstimator},
    convert::point_cloud_with_normals_to_data_matrix,
    filter::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig},
    io::read_point_cloud_file,
    plane::estimate_plane_from_cloud,
    segmentation::{EuclideanClusterConfig, EuclideanClusterExtractor},
    voxelize::{VoxelGridConfig, voxelize},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // ── 1. Load ──────────────────────────────────────────────────────────────
    let raw = if args.len() > 1 {
        println!("Loading: {}", args[1]);
        read_point_cloud_file(&args[1])?
    } else {
        println!("No file supplied — using synthetic scene (plane + objects).");
        synthetic_scene()
    };
    println!("[1] Loaded          {} points", raw.len());

    // ── 2. Voxel downsample ──────────────────────────────────────────────────
    let cloud = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.05)).filter(&raw)?;
    println!("[2] Voxel (5 cm)    {} points", cloud.len());

    // ── 3. Normal estimation (enables NAPSAC in inlier) ──────────────────────
    let (cloud_n, diag) = NormalEstimator::new(NormalEstimationConfig::k_neighbors(16))
        .estimate_with_diagnostics(&cloud)?;
    println!(
        "[3] Normals         ok={} invalid={}",
        diag.valid_count, diag.invalid_count
    );

    // Show that normals are available for neighbourhood-aware sampling
    let _data6 = point_cloud_with_normals_to_data_matrix(&cloud_n)?;
    println!("    DataMatrix [x,y,z,nx,ny,nz] shape: {}×6", cloud_n.len());

    // ── 4. Plane segmentation via inlier's RANSAC ────────────────────────────
    let plane = estimate_plane_from_cloud(&cloud, 0.05, None)?;
    println!(
        "[4] Plane           normal=[{:.3},{:.3},{:.3}] d={:.3}  inliers={}  iters={}",
        plane.normal[0],
        plane.normal[1],
        plane.normal[2],
        plane.d,
        plane.inlier_cloud.len(),
        plane.iterations,
    );

    // ── 5. Cluster the non-ground residuals ──────────────────────────────────
    let clusters = EuclideanClusterExtractor::new(EuclideanClusterConfig::with_tolerance(0.15, 20))
        .extract(&plane.outlier_cloud)?;
    println!(
        "[5] Clusters        {} found in {} outlier points",
        clusters.cluster_count,
        plane.outlier_cloud.len(),
    );
    for (i, &sz) in clusters.cluster_sizes.iter().enumerate() {
        println!("    cluster[{}] {} points", i, sz);
    }

    // ── 6. Occupancy grid of the full scene ──────────────────────────────────
    let grid = voxelize(&cloud, VoxelGridConfig::new(0.10))?;
    println!(
        "[6] Occupancy grid  dims={:?}  occupied={}/{}",
        grid.dims,
        grid.occupied_count(),
        grid.len(),
    );

    println!("\nDone.");
    Ok(())
}

/// Synthetic scene: floor plane + two box-shaped object clusters + noise.
fn synthetic_scene() -> PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    let mut seed: u64 = 0xcafe_babe;

    let mut rng = move || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32)
    };

    // Floor (z ≈ 0)
    for _ in 0..3000 {
        builder
            .push_point([(rng() - 0.5) * 20.0, (rng() - 0.5) * 20.0, rng() * 0.02])
            .unwrap();
    }
    // Object A: box around (3, 2, 0.5)
    for _ in 0..400 {
        builder
            .push_point([3.0 + rng() * 1.0, 2.0 + rng() * 1.0, 0.5 + rng() * 1.0])
            .unwrap();
    }
    // Object B: box around (-4, -3, 0.5)
    for _ in 0..300 {
        builder
            .push_point([-4.0 + rng() * 1.5, -3.0 + rng() * 1.5, 0.5 + rng() * 1.0])
            .unwrap();
    }
    // Random noise
    for _ in 0..200 {
        builder
            .push_point([
                (rng() - 0.5) * 20.0,
                (rng() - 0.5) * 20.0,
                rng() * 4.0 + 1.5,
            ])
            .unwrap();
    }

    builder.build().unwrap()
}
