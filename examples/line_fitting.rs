//! Example: Robust line fitting using RANSAC
//!
//! This example demonstrates robust line fitting using the LineEstimator.

use inlier::api::estimate_line;
use nalgebra::DMatrix;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Robust Line Fitting Example ===\n");

    // Generate synthetic data: y = 2x + 1 with noise, plus outliers
    let n_inliers = 60;
    let n_outliers = 25;
    let n_total = n_inliers + n_outliers;

    let mut rng = rand::thread_rng();

    // True line parameters: y = mx + b
    let true_slope = 2.0;
    let true_intercept = 1.0;

    println!("True line: y = {:.2}x + {:.2}", true_slope, true_intercept);
    println!(
        "Generating {} inliers and {} outliers\n",
        n_inliers, n_outliers
    );

    // Generate inliers along the line with noise
    let mut points = Vec::new();
    for i in 0..n_inliers {
        let x = (i as f64) * 0.2 - 6.0;
        let y = true_slope * x + true_intercept + rng.gen_range(-0.3..0.3);
        points.push((x, y));
    }

    // Generate outliers (random points)
    for _ in 0..n_outliers {
        let x = rng.gen_range(-10.0..10.0);
        let y = rng.gen_range(-20.0..20.0);
        points.push((x, y));
    }

    // Shuffle points
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    points.shuffle(&mut rng);

    // Convert to DMatrix format
    let mut points_matrix = DMatrix::<f64>::zeros(n_total, 2);
    for (i, &(x, y)) in points.iter().enumerate() {
        points_matrix[(i, 0)] = x;
        points_matrix[(i, 1)] = y;
    }

    // Estimate line using RANSAC
    let threshold = 0.5; // Distance threshold
    let result = estimate_line(&points_matrix, threshold, None)?;

    println!("RANSAC Results:");
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

    // Display estimated line
    let line = &result.model;
    let params = line.params();
    println!(
        "\nEstimated line: {:.4}x + {:.4}y + {:.4} = 0",
        params[0], params[1], params[2]
    );

    // Convert to slope-intercept form for comparison
    if let Some((slope, intercept)) = line.to_slope_intercept() {
        println!(
            "  In slope-intercept form: y = {:.4}x + {:.4}",
            slope, intercept
        );
        println!(
            "  True line: y = {:.4}x + {:.4}",
            true_slope, true_intercept
        );
        println!("  Error in slope: {:.4}", (slope - true_slope).abs());
        println!(
            "  Error in intercept: {:.4}",
            (intercept - true_intercept).abs()
        );
    }

    // Count how many true inliers were found
    let mut found_inliers = 0;
    for &idx in &result.inliers {
        // Check if this point was originally an inlier
        // (we shuffled, so we need to check the actual point)
        let x = points_matrix[(idx, 0)];
        let y = points_matrix[(idx, 1)];
        let expected_y = true_slope * x + true_intercept;
        let dist = (y - expected_y).abs();
        if dist < 0.5 {
            found_inliers += 1;
        }
    }
    println!(
        "\nCorrectly identified {} out of {} true inliers",
        found_inliers, n_inliers
    );

    Ok(())
}
