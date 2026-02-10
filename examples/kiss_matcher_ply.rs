//! Example: Full KISS-Matcher pipeline on real point cloud data from PLY files
//!
//! Usage: cargo run --example kiss_matcher_ply --features io

use inlier::io::load_ply;
use inlier::kiss_matcher::{KISSMatcherConfig, kiss_matcher_full_pipeline};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let src_path: String;
    let dst_path: String;

    if args.len() >= 3 {
        src_path = args[1].clone();
        dst_path = args[2].clone();
    } else {
        // Default to 3DMatch sample (real-world data with ~350k points each)
        println!("Usage: {} <src.ply> <dst.ply>", args[0]);
        println!("Using default: 3DMatch sample (~350k points each)");
        println!("Note: This will be downsampled to ~6k points for feature extraction\n");
        src_path =
            "TEASER-plusplus/examples/example_data/3dmatch_sample/cloud_bin_2.ply".to_string();
        dst_path =
            "TEASER-plusplus/examples/example_data/3dmatch_sample/cloud_bin_36.ply".to_string();
    }

    // Load point clouds
    println!("Loading point clouds...");
    let src = load_ply(&src_path)?;
    let dst = load_ply(&dst_path)?;

    println!("  Source: {} points from {}", src.n_points(), src_path);
    println!("  Target: {} points from {}\n", dst.n_points(), dst_path);

    // Configure KISS-Matcher
    let config = KISSMatcherConfig {
        voxel_size: 0.05,
        normal_radius: 0.15, // Larger radius for sparse data
        fpfh_radius: 0.3,    // Larger FPFH radius
        the_linearity: 10.0,
        robin_noise_bound: 0.05,
        solver_noise_bound: 0.01,
        ratio_threshold: 0.95, // Very loose for real-world data
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Voxel size: {}", config.voxel_size);
    println!("  Normal radius: {}", config.normal_radius);
    println!("  FPFH radius: {}", config.fpfh_radius);
    println!("  Ratio threshold: {}", config.ratio_threshold);
    println!("  ROBIN noise bound: {}", config.robin_noise_bound);
    println!();

    // Run KISS-Matcher pipeline
    match kiss_matcher_full_pipeline(&src, &dst, &config) {
        Some(result) => {
            println!("\n✓ Registration succeeded!");
            println!("═══════════════════════════════════════");
            println!("Scale:       {:.6}", result.scale);
            println!("\nRotation:");
            println!("{:.6}", result.rotation);
            println!(
                "Translation: [{:.6}, {:.6}, {:.6}]",
                result.translation[0], result.translation[1], result.translation[2]
            );
            println!("\nStatistics:");
            println!(
                "  Initial correspondences:     {}",
                result.n_correspondences_initial
            );
            println!(
                "  After ROBIN pruning:         {}",
                result.n_correspondences_after_robin
            );
            println!(
                "  Final inliers (after GNC):   {}",
                result.inlier_indices.len()
            );
            println!(
                "  Inlier ratio:                {:.1}%",
                100.0 * result.inlier_indices.len() as f64
                    / result.n_correspondences_initial as f64
            );
            Ok(())
        }
        None => {
            eprintln!("\n✗ Registration failed!");
            eprintln!("Possible reasons:");
            eprintln!("  - Not enough distinctive features");
            eprintln!("  - Point clouds are too sparse or synthetic");
            eprintln!("  - Parameters need tuning");
            eprintln!("\nTry:");
            eprintln!("  - Increasing normal_radius and fpfh_radius");
            eprintln!("  - Using real-world point clouds (not synthetic test data)");
            eprintln!("  - Loosening ratio_threshold to 0.95-0.99");
            eprintln!("\nNote: KISS-Matcher is feature-based and needs distinctive geometry.");
            eprintln!("Synthetic test data (e.g., TEASER benchmark) lacks natural features.");
            Err("Pipeline failed".into())
        }
    }
}
