//! Non-rigid point cloud registration with scale drift
//!
//! Demonstrates RBF-based smooth scale field estimation for registering
//! point clouds with spatially-varying scale (e.g., thermal expansion, growth).
//!
//! Usage:
//!   cargo run --example nonrigid_registration --features io <src.ply> <dst.ply>
//!   cargo run --example nonrigid_registration  # Run synthetic example

use inlier::io::load_ply;

use inlier::estimators::rbf_scale_field::{RBFKernel, RBFScaleConfig};
use inlier::matcher::config::KISSMatcherConfig;
use inlier::matcher::pipeline_nonrigid::{
    FeatureMethod, NonRigidKISSConfig, nonrigid_kiss_matcher_pipeline,
};
use inlier::matcher::sipfh::SIPFHConfig;
use inlier::types::{DataMatrix, Point3};
use nalgebra::{Rotation3, Vector3};
use rand::{Rng, thread_rng};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() >= 3 {
        // PLY file mode
        run_ply_example(&args[1], &args[2])?;
        return Ok(());
    }

    // Synthetic data mode
    println!("═══════════════════════════════════════════════════════════");
    println!("   Non-Rigid Point Cloud Registration with Scale Drift");
    println!("═══════════════════════════════════════════════════════════");
    println!("\n[Synthetic Data Mode]");

    {
        println!("Usage: {} <src.ply> <dst.ply>", args[0]);
        println!("       (to load PLY files instead)\n");
    }

    println!("\nExample: Moderate Scale Variation");
    println!("─────────────────────────────────────────────────────────\n");
    run_synthetic_example();

    Ok(())
}

fn run_ply_example(src_path: &str, dst_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("   Non-Rigid Point Cloud Registration with Scale Drift");
    println!("═══════════════════════════════════════════════════════════");
    println!("\n[PLY File Mode]\n");

    // Load point clouds
    println!("Loading point clouds...");
    let src = load_ply(src_path)?;
    let dst = load_ply(dst_path)?;

    println!("  Source: {} points from {}", src.n_points(), src_path);
    println!("  Target: {} points from {}\n", dst.n_points(), dst_path);

    // Configure non-rigid pipeline with SIPFH for scale-invariance
    let config = NonRigidKISSConfig {
        base: KISSMatcherConfig {
            voxel_size: 0.05,
            normal_radius: 0.15,
            fpfh_radius: 0.3,
            ratio_threshold: 0.9,
            robin_noise_bound: 0.05,
            solver_noise_bound: 0.02,
            the_linearity: 10.0,
            ..Default::default()
        },
        rbf: RBFScaleConfig {
            num_control_points: 50,
            kernel: RBFKernel::Gaussian { sigma: 0.2 },
            regularization_lambda: 1e-3,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            min_scale: 0.8,
            max_scale: 1.5,
            use_sparse: true,
        },
        feature_method: FeatureMethod::SIPFH(SIPFHConfig {
            num_octaves: 3,
            scales_per_octave: 3,
            initial_sigma: 0.05,
            dog_threshold: 0.015,
            edge_threshold: 10.0,
            fpfh_radius: 0.3,
            the_linearity: 0.9,
            fpfh_bins: 11,
            scale_weight: 0.5,
        }),
    };

    println!("Configuration:");
    println!("  Feature method: SIPFH (scale-invariant)");
    println!("  Voxel size: {}", config.base.voxel_size);
    println!("  Normal radius: {}", config.base.normal_radius);
    println!("  FPFH radius: {}", config.base.fpfh_radius);
    println!("  Ratio threshold: {}", config.base.ratio_threshold);
    println!("  ROBIN noise bound: {}", config.base.robin_noise_bound);
    println!("  RBF control points: {}", config.rbf.num_control_points);
    println!();

    // Run non-rigid pipeline
    match nonrigid_kiss_matcher_pipeline(&src, &dst, &config) {
        Some(result) => {
            println!("\n✓ Registration succeeded!");
            print_results(&result);

            // Show scale field statistics
            println!("\nScale Field Analysis:");
            if !result.transform.scale_field.control_points.is_empty() {
                let scales: Vec<f64> = result
                    .transform
                    .scale_field
                    .control_points
                    .iter()
                    .map(|p| result.transform.scale_field.eval(p))
                    .collect();

                let min_scale = scales.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_scale = scales.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                println!("  Min scale: {min_scale:.6}");
                println!("  Max scale: {max_scale:.6}");
                println!(
                    "  Scale range: {:.6} ({:.1}% variation)",
                    max_scale - min_scale,
                    100.0 * (max_scale - min_scale) / result.mean_scale
                );
            }
        }
        None => {
            println!("\n✗ Registration failed!");
            println!("\nPossible reasons:");
            println!("  - Point clouds too different (large transformation)");
            println!("  - Too few distinctive features");
            println!("  - Large non-rigid deformation (>20% scale variation)");
            println!("\nTips:");
            println!("  - Try increasing normal_radius and fpfh_radius");
            println!("  - Use denser point clouds or reduce voxel_size");
            println!("  - Ensure point clouds have sufficient overlap");
        }
    }

    Ok(())
}

fn run_synthetic_example() {
    // Generate realistic irregular point cloud with more structure
    let mut rng = thread_rng();
    let mut src_points = Vec::new();

    // Create a semi-structured point cloud (grid with jitter)
    for xi in 0..15 {
        for yi in 0..15 {
            for zi in 0..8 {
                let x = xi as f64 + rng.gen_range(-0.3..0.3);
                let y = yi as f64 + rng.gen_range(-0.3..0.3);
                let z = zi as f64 + rng.gen_range(-0.3..0.3);
                src_points.push(Point3::new(x, y, z));
            }
        }
    }

    println!(
        "Source cloud: {} points (semi-structured)",
        src_points.len()
    );
    println!("Scale model: s(x) = 1.0 + 0.01*x (varies from 1.0 to 1.15)");
    println!("Note: Very small scale variation to ensure features still match\n");

    // Apply spatially-varying transformation with SMALL scale variation
    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.15);
    let translation = Vector3::new(1.0, 0.5, 0.2);

    let dst_points: Vec<Point3> = src_points
        .iter()
        .map(|p| {
            // Very small scale variation: 1.0 to 1.15
            let local_scale = 1.0 + 0.01 * p.x;
            local_scale * (rotation * p) + translation
        })
        .collect();

    // Add small noise
    let mut rng2 = thread_rng();
    let dst_points_noisy: Vec<Point3> = dst_points
        .iter()
        .map(|p| {
            let noise = Vector3::new(
                rng2.gen_range(-0.015..0.015),
                rng2.gen_range(-0.015..0.015),
                rng2.gen_range(-0.015..0.015),
            );
            p + noise
        })
        .collect();

    // Convert to DataMatrix
    let src_matrix = points_to_matrix(&src_points);
    let dst_matrix = points_to_matrix(&dst_points_noisy);

    // Configure non-rigid pipeline with FasterPFH for synthetic (already aligned)
    let config = NonRigidKISSConfig {
        base: KISSMatcherConfig {
            voxel_size: 0.8,
            normal_radius: 1.5,
            fpfh_radius: 3.0,
            ratio_threshold: 0.85,
            robin_noise_bound: 0.2,
            solver_noise_bound: 0.15,
            ..Default::default()
        },
        rbf: RBFScaleConfig {
            num_control_points: 40,
            kernel: RBFKernel::Gaussian { sigma: 3.0 },
            regularization_lambda: 5e-4,
            max_iterations: 80,
            convergence_threshold: 1e-4,
            min_scale: 0.9,
            max_scale: 1.3,
            use_sparse: true,
        },
        feature_method: FeatureMethod::FasterPFH, // Synthetic data doesn't need scale-invariance
    };

    // Run pipeline
    match nonrigid_kiss_matcher_pipeline(&src_matrix, &dst_matrix, &config) {
        Some(result) => {
            println!("\n✓ Registration succeeded!\n");
            print_results(&result);

            // Verify scale variation
            println!("\nScale Verification:");
            let scale_at_x0 = result
                .transform
                .scale_field
                .eval(&Point3::new(0.0, 7.0, 4.0));
            let scale_at_x7 = result
                .transform
                .scale_field
                .eval(&Point3::new(7.0, 7.0, 4.0));
            let scale_at_x14 = result
                .transform
                .scale_field
                .eval(&Point3::new(14.0, 7.0, 4.0));

            println!("  Scale at x=0:  {scale_at_x0:.4} (expected ~1.00)");
            println!("  Scale at x=7:  {scale_at_x7:.4} (expected ~1.07)");
            println!("  Scale at x=14: {scale_at_x14:.4} (expected ~1.14)");

            // Compute and show residuals
            let residuals: Vec<f64> = src_points
                .iter()
                .zip(dst_points_noisy.iter())
                .map(|(s, d)| result.transform.residual(s, d))
                .collect();

            let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
            let max_residual = residuals.iter().cloned().fold(0.0, f64::max);

            println!("\nResidual Statistics:");
            println!("  Mean residual: {mean_residual:.6}");
            println!("  Max residual:  {max_residual:.6}");
        }
        None => {
            println!("✗ Registration failed!");
            println!("\nNote: Non-rigid matching is challenging with FPFH features.");
            println!("      FPFH descriptors are not fully scale-invariant at local level.");
        }
    }
}

// Helper functions

fn points_to_matrix(points: &[Point3]) -> DataMatrix {
    let n = points.len();
    let mut matrix = DataMatrix::zeros(n, 3);
    for (i, p) in points.iter().enumerate() {
        matrix.set(i, 0, p.x);
        matrix.set(i, 1, p.y);
        matrix.set(i, 2, p.z);
    }
    matrix
}

fn print_results(result: &inlier::matcher::pipeline_nonrigid::NonRigidKISSResult) {
    println!("═══════════════════════════════════════");
    println!("Scale Field:");
    println!("  Mean scale:       {:.6}", result.mean_scale);
    println!("  Scale std dev:    {:.6}", result.scale_std);
    println!(
        "  Control points:   {}",
        result.transform.scale_field.control_points.len()
    );

    println!("\nRotation:");
    for row in 0..3 {
        print!("  ");
        for col in 0..3 {
            print!(" {:8.5}", result.transform.rotation[(row, col)]);
        }
        println!();
    }

    println!(
        "\nTranslation: [{:.6}, {:.6}, {:.6}]",
        result.transform.translation.x,
        result.transform.translation.y,
        result.transform.translation.z
    );

    println!("\nCorrespondence Statistics:");
    println!(
        "  Initial correspondences:     {}",
        result.n_correspondences_initial
    );
    println!(
        "  After ROBIN pruning:         {}",
        result.n_correspondences_after_robin
    );
    println!(
        "  Final inliers:               {}",
        result.n_correspondences_final
    );
    println!(
        "  Inlier ratio:                {:.1}%",
        100.0 * result.n_correspondences_final as f64 / result.n_correspondences_initial as f64
    );
}
