//! Example: Homography estimation from point correspondences
//!
//! This example demonstrates homography estimation using RANSAC
//! on synthetic 2D point correspondences.

use inlier::*;
use nalgebra::DMatrix;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Homography Estimation Example ===\n");

    // Generate synthetic data: points on a plane with a known transformation
    let n_points = 30;
    let n_outliers = 10;
    let n_total = n_points + n_outliers;

    let mut rng = rand::thread_rng();
    let mut points1 = DMatrix::<f64>::zeros(n_total, 2);
    let mut points2 = DMatrix::<f64>::zeros(n_total, 2);

    // Generate inliers: points transformed by a known homography
    // Simple case: translation + slight rotation
    let angle: f64 = 0.1; // Small rotation
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let tx = 10.0;
    let ty = 5.0;

    for i in 0..n_points {
        let x = (i as f64) * 5.0 - 50.0;
        let y = (i as f64) * 3.0 - 30.0;

        points1[(i, 0)] = x;
        points1[(i, 1)] = y;

        // Apply transformation: rotation + translation
        let x_rot = cos_a * x - sin_a * y;
        let y_rot = sin_a * x + cos_a * y;
        points2[(i, 0)] = x_rot + tx + rng.gen_range(-0.5..0.5); // Add noise
        points2[(i, 1)] = y_rot + ty + rng.gen_range(-0.5..0.5);
    }

    // Generate outliers (random correspondences)
    for i in n_points..n_total {
        points1[(i, 0)] = rng.gen_range(-100.0..100.0);
        points1[(i, 1)] = rng.gen_range(-100.0..100.0);
        points2[(i, 0)] = rng.gen_range(-100.0..100.0);
        points2[(i, 1)] = rng.gen_range(-100.0..100.0);
    }

    println!("Generated {} inliers and {} outliers", n_points, n_outliers);
    println!("True transformation: rotation ({:.2} rad) + translation ({:.1}, {:.1})\n",
             angle, tx, ty);

    // Estimate homography
    let threshold = 2.0; // pixels
    let result = estimate_homography(&points1, &points2, threshold, None)?;

    println!("Estimation results:");
    println!("  Found {} inliers out of {} points", result.inliers.len(), n_total);
    println!("  Inlier ratio: {:.2}%",
             100.0 * result.inliers.len() as f64 / n_total as f64);
    println!("  Score: {:?}", result.score);
    println!("  Iterations: {}", result.iterations);

    println!("\nEstimated homography matrix:");
    for i in 0..3 {
        println!("  [{:8.4}, {:8.4}, {:8.4}]",
                 result.model.h[(i, 0)],
                 result.model.h[(i, 1)],
                 result.model.h[(i, 2)]);
    }

    // Verify inliers
    let mut correct_inliers = 0;
    for &idx in &result.inliers {
        if idx < n_points {
            correct_inliers += 1;
        }
    }
    println!("\nCorrectly identified {} out of {} true inliers",
             correct_inliers, n_points);

    Ok(())
}
