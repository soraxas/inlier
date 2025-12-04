//! Example: Robust linear regression using RANSAC
//!
//! This example demonstrates how to use the inlier library to perform
//! robust linear regression on 2D data with outliers using the LineEstimator.
//! The results are visualized and saved to a PNG image.

use inlier::api::estimate_line;
use nalgebra::DMatrix;
use plotters::prelude::*;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Robust Linear Regression Example ===\n");

    // Generate synthetic data: y = 2x + 1 with noise, plus outliers
    let n_inliers = 50;
    let n_outliers = 20;
    let n_total = n_inliers + n_outliers;

    let mut rng = rand::rng();
    let mut points = DMatrix::<f64>::zeros(n_total, 2);

    // True line parameters: y = mx + b
    let true_slope = 2.0;
    let true_intercept = 1.0;

    // Generate inliers: y = 2x + 1 + noise
    for i in 0..n_inliers {
        let x = (i as f64) * 0.1 - 2.5;
        let y = true_slope * x + true_intercept + rng.random_range(-0.1..0.1);
        points[(i, 0)] = x;
        points[(i, 1)] = y;
    }

    // Generate outliers (random points)
    for i in n_inliers..n_total {
        points[(i, 0)] = rng.random_range(-5.0..5.0);
        points[(i, 1)] = rng.random_range(-10.0..10.0);
    }

    // Shuffle points to mix inliers and outliers
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_total).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_points = DMatrix::<f64>::zeros(n_total, 2);
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        shuffled_points[(new_idx, 0)] = points[(old_idx, 0)];
        shuffled_points[(new_idx, 1)] = points[(old_idx, 1)];
    }

    println!(
        "Generated {} inliers and {} outliers",
        n_inliers, n_outliers
    );
    println!(
        "True line: y = {:.2}x + {:.2}\n",
        true_slope, true_intercept
    );

    // Estimate line using RANSAC
    let threshold = 0.2; // Distance threshold
    let result = estimate_line(&shuffled_points, threshold, None)?;

    println!("RANSAC Linear Regression Results:");
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
        let x = shuffled_points[(idx, 0)];
        let y = shuffled_points[(idx, 1)];
        let expected_y = true_slope * x + true_intercept;
        let dist = (y - expected_y).abs();
        if dist < 0.2 {
            found_inliers += 1;
        }
    }
    println!(
        "\nCorrectly identified {} out of {} true inliers",
        found_inliers, n_inliers
    );

    // Create plot and save to image
    let output_file = "linear_regression_result.png";
    let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Robust Linear Regression with RANSAC",
            ("sans-serif", 40).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(-6.0..6.0, -12.0..12.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Plot all points, coloring inliers and outliers differently
    let inlier_set: std::collections::HashSet<usize> = result.inliers.iter().cloned().collect();

    // Plot outliers (red)
    for i in 0..n_total {
        if !inlier_set.contains(&i) {
            chart
                .draw_series(std::iter::once(Circle::new(
                    (shuffled_points[(i, 0)], shuffled_points[(i, 1)]),
                    3,
                    RED.filled(),
                )))
                .unwrap();
        }
    }

    // Plot inliers (green)
    for &idx in &result.inliers {
        chart
            .draw_series(std::iter::once(Circle::new(
                (shuffled_points[(idx, 0)], shuffled_points[(idx, 1)]),
                3,
                GREEN.filled(),
            )))
            .unwrap();
    }

    // Plot true line (blue, dashed)
    let true_line_points: Vec<(f64, f64)> = (0..=120)
        .map(|i| {
            let x = -6.0 + (i as f64) * 0.1;
            (x, true_slope * x + true_intercept)
        })
        .collect();
    chart
        .draw_series(LineSeries::new(true_line_points, BLUE.stroke_width(2)))
        .unwrap()
        .label("True line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    // Plot estimated line (red, solid)
    if let Some((slope, intercept)) = line.to_slope_intercept() {
        let estimated_line_points: Vec<(f64, f64)> = (0..=120)
            .map(|i| {
                let x = -6.0 + (i as f64) * 0.1;
                (x, slope * x + intercept)
            })
            .collect();
        chart
            .draw_series(LineSeries::new(estimated_line_points, RED.stroke_width(2)))
            .unwrap()
            .label("Estimated line")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("\nPlot saved to: {}", output_file);

    Ok(())
}
