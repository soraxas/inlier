//! Demonstrates RBF-based smooth scale field estimation
//!
//! Shows how spatially-varying scales can be estimated using
//! radial basis function interpolation with sparse control points.

use inlier::estimators::{RBFKernel, RBFScaleConfig, RBFScaleEstimator};
use inlier::types::Point3;
use nalgebra::{Rotation3, Vector3};

fn main() {
    println!("=== RBF Scale Field Estimator Demo ===\n");

    // Example 1: Uniform scale
    println!("[Example 1] Uniform scale transformation");
    uniform_scale_example();

    println!("\n{}\n", "=".repeat(60));

    // Example 2: Spatially-varying scale (linear gradient)
    println!("[Example 2] Linear scale gradient");
    linear_gradient_example();

    println!("\n{}\n", "=".repeat(60));

    // Example 3: Radial expansion
    println!("[Example 3] Radial expansion");
    radial_expansion_example();
}

fn uniform_scale_example() {
    // Generate source points
    let mut src = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            src.push(Point3::new(x as f64, y as f64, 0.5));
        }
    }

    // Apply uniform scale + rotation + translation
    let scale = 1.5;
    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.2);
    let translation = Vector3::new(1.0, 2.0, 0.5);

    let dst: Vec<Point3> = src
        .iter()
        .map(|p| scale * (rotation * p) + translation)
        .collect();

    println!("  Source points: {}", src.len());
    println!("  True scale: {scale}");

    // Estimate transformation
    let config = RBFScaleConfig {
        num_control_points: 20,
        max_iterations: 50,
        regularization_lambda: 1e-3,
        kernel: RBFKernel::Gaussian { sigma: 1.5 },
        ..Default::default()
    };

    let estimator = RBFScaleEstimator::new(config);
    let result = estimator.estimate(&src, &dst).unwrap();

    println!("  Estimated mean scale: {:.3}", result.mean_scale());
    println!(
        "  Control points: {}",
        result.scale_field.control_points.len()
    );

    // Compute residuals
    let residuals: Vec<f64> = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| result.residual(s, d))
        .collect();

    let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let max_residual = residuals.iter().cloned().fold(0.0, f64::max);

    println!("  Mean residual: {mean_residual:.6}");
    println!("  Max residual: {max_residual:.6}");
}

fn linear_gradient_example() {
    // Generate source points in a grid
    let mut src = Vec::new();
    for x in 0..8 {
        for y in 0..8 {
            src.push(Point3::new(x as f64, y as f64, 0.0));
        }
    }

    // Apply spatially-varying scale: s(x) = 1.0 + 0.1*x
    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.1);
    let translation = Vector3::new(0.5, 0.5, 0.0);

    let dst: Vec<Point3> = src
        .iter()
        .map(|p| {
            let local_scale = 1.0 + 0.1 * p.x;
            local_scale * (rotation * p) + translation
        })
        .collect();

    println!("  Source points: {}", src.len());
    println!("  Scale varies: 1.0 at x=0 to 1.7 at x=7");

    // Estimate transformation
    let config = RBFScaleConfig {
        num_control_points: 30,
        max_iterations: 100,
        regularization_lambda: 1e-4,
        kernel: RBFKernel::Gaussian { sigma: 2.0 },
        ..Default::default()
    };

    let estimator = RBFScaleEstimator::new(config);
    let result = estimator.estimate(&src, &dst).unwrap();

    println!("  Estimated mean scale: {:.3}", result.mean_scale());
    println!(
        "  Control points: {}",
        result.scale_field.control_points.len()
    );

    // Check scale variation
    let scale_at_x0 = result.scale_field.eval(&Point3::new(0.0, 3.5, 0.0));
    let scale_at_x3 = result.scale_field.eval(&Point3::new(3.5, 3.5, 0.0));
    let scale_at_x7 = result.scale_field.eval(&Point3::new(7.0, 3.5, 0.0));

    println!("  Scale at x=0: {scale_at_x0:.3}");
    println!("  Scale at x=3.5: {scale_at_x3:.3}");
    println!("  Scale at x=7: {scale_at_x7:.3}");

    // Compute residuals
    let residuals: Vec<f64> = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| result.residual(s, d))
        .collect();

    let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let max_residual = residuals.iter().cloned().fold(0.0, f64::max);

    println!("  Mean residual: {mean_residual:.6}");
    println!("  Max residual: {max_residual:.6}");
}

fn radial_expansion_example() {
    // Generate source points in a circle
    let mut src = Vec::new();
    for i in 0..40 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 40.0;
        let r = 2.0;
        src.push(Point3::new(r * angle.cos(), r * angle.sin(), 0.0));
    }

    // Apply radial expansion: s(r) = 1.0 + 0.3 * r/R
    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.15);
    let translation = Vector3::new(0.2, 0.1, 0.0);

    let dst: Vec<Point3> = src
        .iter()
        .map(|p| {
            let r = (p.x * p.x + p.y * p.y).sqrt();
            let local_scale = 1.0 + 0.3 * r / 2.0;
            local_scale * (rotation * p) + translation
        })
        .collect();

    println!("  Source points: {}", src.len());
    println!("  Scale varies radially: 1.0 at center to 1.3 at edge");

    // Estimate transformation
    let config = RBFScaleConfig {
        num_control_points: 25,
        max_iterations: 100,
        regularization_lambda: 5e-4,
        kernel: RBFKernel::Gaussian { sigma: 1.0 },
        ..Default::default()
    };

    let estimator = RBFScaleEstimator::new(config);
    let result = estimator.estimate(&src, &dst).unwrap();

    println!("  Estimated mean scale: {:.3}", result.mean_scale());
    println!(
        "  Control points: {}",
        result.scale_field.control_points.len()
    );

    // Check scale variation
    let scale_at_center = result.scale_field.eval(&Point3::new(0.0, 0.0, 0.0));
    let scale_at_edge = result.scale_field.eval(&Point3::new(2.0, 0.0, 0.0));

    println!("  Scale at center: {scale_at_center:.3}");
    println!("  Scale at edge: {scale_at_edge:.3}");

    // Compute residuals
    let residuals: Vec<f64> = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| result.residual(s, d))
        .collect();

    let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let max_residual = residuals.iter().cloned().fold(0.0, f64::max);

    println!("  Mean residual: {mean_residual:.6}");
    println!("  Max residual: {max_residual:.6}");
}
