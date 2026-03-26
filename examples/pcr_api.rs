//! Example: Point Cloud Registration using the high-level PCR API
//!
//! This example demonstrates the simplified API for point cloud registration
//! with both rigid and non-rigid transformations.

use inlier::io::load_ply;
use inlier::pcr::{PCRConfig, register_nonrigid, register_rigid};
use nalgebra::{Matrix3, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!(
            "Usage: cargo run --example pcr_api --features io <source.ply> <target.ply> [nonrigid]"
        );
        println!("\nThis example demonstrates the high-level PCR (Point Cloud Registration) API.");
        println!("The API provides simple functions for rigid and non-rigid registration:");
        println!("  - register_rigid(): Rotation + translation + uniform scale");
        println!("  - register_nonrigid(): Spatially-varying scale + rigid (using SIPFH + RBF)");
        std::process::exit(1);
    }

    let src_path = &args[1];
    let dst_path = &args[2];
    let use_nonrigid = args.len() > 3 && args[3] == "nonrigid";

    println!("\n{}", "=".repeat(70));
    println!("  Point Cloud Registration API Example");
    println!("{}", "=".repeat(70));

    // Load point clouds
    println!("\n[1/3] Loading point clouds...");
    let src = load_ply(src_path)?;
    let dst = load_ply(dst_path)?;
    println!("  Source: {} points", src.n_points());
    println!("  Target: {} points", dst.n_points());

    // Configure registration
    let base_config = PCRConfig {
        voxel_size: 0.05,
        feature_radius: 0.3,
        normal_radius: 0.15,
        ..PCRConfig::default()
    };

    if use_nonrigid {
        // Non-rigid registration with scale-invariant features
        let config = PCRConfig {
            use_scale_invariant_features: true,
            ..base_config.clone()
        };

        println!("\n[2/3] Running non-rigid registration (SIPFH + RBF)...");
        println!("  Config: {config:?}");

        let result = register_nonrigid(&src, &dst, &config);

        println!("\n[3/3] Results:");
        println!("{}", "=".repeat(70));

        if let Some(r) = result {
            println!("✓ Registration succeeded!\n");
            print_rotation(&r.transform.rotation);
            print_translation(&r.transform.translation);
            println!("\nScale Field:");
            println!("  Mean scale:       {:.6}", r.mean_scale);
            println!(
                "  Scale std dev:    {:.6} ({:.1}% variation)",
                r.scale_std,
                100.0 * r.scale_std / r.mean_scale
            );

            println!("\nCorrespondences:");
            println!("  Initial:          {}", r.total_correspondences);
            println!("  Final inliers:    {}", r.inlier_count);
            println!(
                "  Inlier ratio:     {:.1}%",
                100.0 * r.inlier_count as f64 / r.total_correspondences as f64
            );

            if r.scale_std / r.mean_scale > 0.15 {
                println!(
                    "\nNote: Large scale variation ({:.1}%) indicates significant",
                    100.0 * r.scale_std / r.mean_scale
                );
                println!("      non-rigid deformation or potential overfitting.");
            }
        } else {
            println!("✗ Registration failed!\n");
            println!("Possible reasons:");
            println!("  - Too few distinctive features");
            println!("  - Large transformation between point clouds");
            println!("  - Insufficient overlap");
            println!("\nTry adjusting parameters:");
            println!("  - Increase feature_radius for more distinctive features");
            println!("  - Decrease voxel_size for denser sampling");
            println!("  - Ensure point clouds have >50% overlap");
        }
    } else {
        // Rigid registration with FasterPFH
        let config = base_config;

        println!("\n[2/3] Running rigid registration (FasterPFH + KISS-Matcher)...");
        println!("  Config: {config:?}");

        let result = register_rigid(&src, &dst, &config);

        println!("\n[3/3] Results:");
        println!("{}", "=".repeat(70));

        if let Some(r) = result {
            println!("✓ Registration succeeded!\n");
            print_rotation(&r.rotation);
            print_translation(&r.translation);
            println!("\nScale: {:.6}", r.scale);

            println!("\nCorrespondences:");
            println!("  Initial:          {}", r.total_correspondences);
            println!("  Final inliers:    {}", r.inlier_count);
            println!(
                "  Inlier ratio:     {:.1}%",
                100.0 * r.inlier_count as f64 / r.total_correspondences as f64
            );
        } else {
            println!("✗ Registration failed!\n");
            println!("Possible reasons:");
            println!("  - Too few distinctive features");
            println!("  - Large transformation between point clouds");
            println!("  - Non-rigid deformation (try 'nonrigid' mode)");
            println!("\nTry adjusting parameters:");
            println!("  - Increase feature_radius for more matches");
            println!("  - Decrease voxel_size for denser sampling");
        }
    }

    println!("{}", "=".repeat(70));
    Ok(())
}

fn print_rotation(r: &Matrix3<f64>) {
    println!("Rotation:");
    for i in 0..3 {
        print!(" ");
        for j in 0..3 {
            print!(" {:9.6}", r[(i, j)]);
        }
        println!();
    }
}

fn print_translation(t: &Vector3<f64>) {
    println!("\nTranslation: [{:.6}, {:.6}, {:.6}]", t[0], t[1], t[2]);
}
