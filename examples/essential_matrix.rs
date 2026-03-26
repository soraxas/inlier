//! Example: Essential matrix estimation from calibrated correspondences
//!
//! This example simulates two calibrated camera views, injects tentative
//! outliers, and estimates the essential matrix with RANSAC.

use inlier::{estimate_essential_matrix, settings::MetasacSettings, types::DataMatrix};
use nalgebra::{UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

#[derive(Clone, Debug)]
struct Correspondence {
    src: (f64, f64),
    dst: (f64, f64),
    expected_inlier: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Essential Matrix Estimation Example ===\n");

    let n_inliers = 60;
    let n_outliers = 18;
    let mut rng = StdRng::seed_from_u64(0xE551_EE55);

    let rotation_gt = UnitQuaternion::from_euler_angles(0.03, -0.08, 0.02).to_rotation_matrix();
    let translation_gt = Vector3::new(0.25, -0.04, 0.03);

    let mut correspondences = Vec::with_capacity(n_inliers + n_outliers);
    for _ in 0..n_inliers {
        let point = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-0.8..0.8),
            rng.random_range(3.5..6.5),
        );

        let cam1 = project(point);
        let cam2 = project(rotation_gt * point + translation_gt);
        correspondences.push(Correspondence {
            src: (
                cam1.0 + rng.random_range(-0.0015..0.0015),
                cam1.1 + rng.random_range(-0.0015..0.0015),
            ),
            dst: (
                cam2.0 + rng.random_range(-0.0015..0.0015),
                cam2.1 + rng.random_range(-0.0015..0.0015),
            ),
            expected_inlier: true,
        });
    }

    for _ in 0..n_outliers {
        correspondences.push(Correspondence {
            src: (rng.random_range(-0.45..0.45), rng.random_range(-0.35..0.35)),
            dst: (rng.random_range(-0.45..0.45), rng.random_range(-0.35..0.35)),
            expected_inlier: false,
        });
    }

    correspondences.shuffle(&mut rng);

    let mut points1 = DataMatrix::zeros(correspondences.len(), 2);
    let mut points2 = DataMatrix::zeros(correspondences.len(), 2);
    for (idx, corr) in correspondences.iter().enumerate() {
        points1.set(idx, 0, corr.src.0);
        points1.set(idx, 1, corr.src.1);
        points2.set(idx, 0, corr.dst.0);
        points2.set(idx, 1, corr.dst.1);
    }

    let settings = MetasacSettings {
        min_iterations: 500,
        max_iterations: 5_000,
        confidence: 0.999,
        ..MetasacSettings::default()
    };
    let threshold = 0.005;
    let result = estimate_essential_matrix(&points1, &points2, threshold, Some(settings))?;

    let expected_inliers = correspondences.iter().filter(|c| c.expected_inlier).count();
    let recovered_true_inliers = result
        .inliers
        .iter()
        .filter(|&&idx| correspondences[idx].expected_inlier)
        .count();

    println!("Generated {expected_inliers} calibrated inliers and {n_outliers} tentative outliers");
    println!(
        "Recovered {} inliers out of {} correspondences",
        result.inliers.len(),
        correspondences.len()
    );
    println!("Correctly kept {recovered_true_inliers} / {expected_inliers} true inliers");
    println!("Iterations: {}", result.iterations);
    println!("Score: {:?}", result.score);

    println!("\nEstimated essential matrix:");
    for row in 0..3 {
        println!(
            "  [{:9.5}, {:9.5}, {:9.5}]",
            result.model.e[(row, 0)],
            result.model.e[(row, 1)],
            result.model.e[(row, 2)]
        );
    }

    let svd = result.model.e.svd(false, false);
    println!(
        "\nSingular values: [{:.5}, {:.5}, {:.5}]",
        svd.singular_values[0], svd.singular_values[1], svd.singular_values[2]
    );
    println!(
        "Ground-truth baseline direction: [{:.3}, {:.3}, {:.3}]",
        translation_gt.normalize().x,
        translation_gt.normalize().y,
        translation_gt.normalize().z
    );

    Ok(())
}

fn project(point: Vector3<f64>) -> (f64, f64) {
    (point.x / point.z, point.y / point.z)
}
