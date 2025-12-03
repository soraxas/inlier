//! Example: Fundamental matrix estimation from point correspondences
//!
//! This example demonstrates fundamental matrix estimation using RANSAC
//! on synthetic 2D point correspondences simulating two camera views.

use inlier::*;
use nalgebra::DMatrix;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fundamental Matrix Estimation Example ===\n");

    // Generate synthetic data: points on a 3D plane viewed from two cameras
    let n_points = 40;
    let n_outliers = 15;
    let n_total = n_points + n_outliers;

    let mut rng = rand::thread_rng();
    let mut points1 = DMatrix::<f64>::zeros(n_total, 2);
    let mut points2 = DMatrix::<f64>::zeros(n_total, 2);

    // Generate inliers: 3D points on a plane, projected to two views
    // Simple case: points on z=10 plane, cameras at different positions
    for i in 0..n_points {
        // 3D point on a plane
        let x_3d = (i as f64) * 2.0 - 40.0;
        let y_3d = (i as f64) * 1.5 - 30.0;
        let z_3d = 10.0;

        // Project to first camera (at origin, looking along +z)
        let x1 = x_3d / z_3d;
        let y1 = y_3d / z_3d;
        points1[(i, 0)] = x1 + rng.gen_range(-0.01..0.01); // Add noise
        points1[(i, 1)] = y1 + rng.gen_range(-0.01..0.01);

        // Project to second camera (translated and slightly rotated)
        let x2_3d = x_3d - 5.0; // Camera translation
        let y2_3d = y_3d - 2.0;
        let x2 = x2_3d / z_3d;
        let y2 = y2_3d / z_3d;
        points2[(i, 0)] = x2 + rng.gen_range(-0.01..0.01);
        points2[(i, 1)] = y2 + rng.gen_range(-0.01..0.01);
    }

    // Generate outliers (random correspondences)
    for i in n_points..n_total {
        points1[(i, 0)] = rng.gen_range(-5.0..5.0);
        points1[(i, 1)] = rng.gen_range(-5.0..5.0);
        points2[(i, 0)] = rng.gen_range(-5.0..5.0);
        points2[(i, 1)] = rng.gen_range(-5.0..5.0);
    }

    println!("Generated {} inliers and {} outliers", n_points, n_outliers);
    println!("Simulating two camera views of points on a plane\n");

    // Estimate fundamental matrix
    let threshold = 0.1; // pixels (normalized coordinates)
    let result = estimate_fundamental_matrix(&points1, &points2, threshold, None)?;

    println!("Estimation results:");
    println!(
        "  Found {} inliers out of {} points",
        result.inliers.len(),
        n_total
    );
    println!(
        "  Inlier ratio: {:.2}%",
        100.0 * result.inliers.len() as f64 / n_total as f64
    );
    println!("  Score: {:?}", result.score);
    println!("  Iterations: {}", result.iterations);

    println!("\nEstimated fundamental matrix:");
    for i in 0..3 {
        println!(
            "  [{:8.4}, {:8.4}, {:8.4}]",
            result.model.f[(i, 0)],
            result.model.f[(i, 1)],
            result.model.f[(i, 2)]
        );
    }

    // Check rank-2 constraint (determinant should be small)
    let det = result.model.f.determinant();
    println!(
        "\nFundamental matrix determinant: {:.6} (should be close to 0)",
        det
    );

    // Verify inliers
    let mut correct_inliers = 0;
    for &idx in &result.inliers {
        if idx < n_points {
            correct_inliers += 1;
        }
    }
    println!(
        "Correctly identified {} out of {} true inliers",
        correct_inliers, n_points
    );

    Ok(())
}
