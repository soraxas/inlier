//! SIPFH (Scale-Invariant Point Feature Histogram) demonstration
//!
//! Shows scale-invariant keypoint detection and matching with SIPFH.

use inlier::matcher::sipfh::{SIPFH, SIPFHConfig};
use inlier::types::DataMatrix;
use nalgebra::{Matrix3, Vector3};
use rand::Rng;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("   SIPFH: Scale-Invariant Point Feature Histogram Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create a simple 3D structure
    let points = create_structured_cloud(500);

    println!("Created point cloud: {} points\n", points.n_points());

    // Configure SIPFH
    let config = SIPFHConfig {
        num_octaves: 3,
        scales_per_octave: 3,
        initial_sigma: 0.05,
        dog_threshold: 0.015,
        edge_threshold: 10.0,
        fpfh_radius: 0.2,
        the_linearity: 0.9,
        fpfh_bins: 11,
        scale_weight: 0.5,
    };

    println!("SIPFH Configuration:");
    println!("  Octaves: {}", config.num_octaves);
    println!("  Scales per octave: {}", config.scales_per_octave);
    println!("  Initial sigma: {}", config.initial_sigma);
    println!("  DoG threshold: {}", config.dog_threshold);
    println!("  FPFH radius: {}", config.fpfh_radius);
    println!("  Scale weight: {}\n", config.scale_weight);

    // Extract SIPFH features
    println!("Extracting scale-invariant keypoints and features...\n");

    let sipfh = SIPFH::new(config);
    let features = sipfh.extract_features(&points);

    println!("✓ Extracted {} scale-invariant features\n", features.len());

    if !features.is_empty() {
        println!("Sample features:");
        for (i, feat) in features.iter().take(5).enumerate() {
            println!(
                "  Feature {}: pos=({:.2}, {:.2}, {:.2}), scale={:.4}, descriptor_dim={}",
                i + 1,
                feat.feature.point.x,
                feat.feature.point.y,
                feat.feature.point.z,
                feat.scale,
                feat.sipfh_descriptor.len()
            );
        }

        println!("\n═══════════════════════════════════════════════════════════");
        println!("Scale-Invariance Test:");
        println!("═══════════════════════════════════════════════════════════\n");

        // Create scaled version
        println!("Creating scaled version (scale=1.5)...");
        let scaled_points = scale_point_cloud(&points, 1.5);

        println!("Extracting features from scaled cloud...\n");
        let scaled_features = sipfh.extract_features(&scaled_points);

        println!("Original cloud: {} features", features.len());
        println!("Scaled cloud:   {} features", scaled_features.len());

        if !features.is_empty() && !scaled_features.is_empty() {
            println!("\n✓ SIPFH successfully extracts keypoints at different scales!");
            println!("  The scale field encodes the transformation, enabling matching");
            println!("  between point clouds with varying scales.");
        }
    } else {
        println!("⚠ No keypoints detected. Try adjusting DoG threshold or point density.");
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Key Advantages of SIPFH:");
    println!("  • Scale-invariant: Matches clouds with different scales");
    println!("  • Fast: Faster than 3D-SIFT (FPFH descriptors)");
    println!("  • Robust: DoG pyramid + edge filtering");
    println!("  • Sparse: Features only at keypoints (10-100× fewer)");
    println!("═══════════════════════════════════════════════════════════");
}

/// Create a structured point cloud
fn create_structured_cloud(n: usize) -> DataMatrix {
    let mut rng = rand::rng();
    let mut points = Vec::new();

    // Create a box-like structure with some randomness
    for _ in 0..n {
        let x = rng.random_range(-1.0..1.0);
        let y = rng.random_range(-1.0..1.0);
        let z = rng.random_range(-1.0..1.0);

        // Add some structure (planes at edges)
        let (x, y, z) = if rng.random_range(0.0..1.0) > 0.7 {
            // Create planar structures
            if rng.random_range(0.0..1.0) > 0.5 {
                (1.0, y, z) // Right plane
            } else {
                (x, 1.0, z) // Top plane
            }
        } else {
            (x, y, z)
        };

        points.push(x);
        points.push(y);
        points.push(z);
    }

    DataMatrix::from_row_slice(n, 3, &points)
}

/// Scale a point cloud uniformly
fn scale_point_cloud(points: &DataMatrix, scale: f64) -> DataMatrix {
    let n = points.n_points();
    let mut scaled = Vec::new();

    for i in 0..n {
        scaled.push(points.get(i, 0) * scale);
        scaled.push(points.get(i, 1) * scale);
        scaled.push(points.get(i, 2) * scale);
    }

    DataMatrix::from_row_slice(n, 3, &scaled)
}
