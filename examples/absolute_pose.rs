//! Example: Absolute pose estimation from 3D-2D correspondences
//!
//! This example simulates a calibrated camera observing 3D landmarks, injects a
//! handful of wrong matches, and estimates the camera pose with RANSAC.

use inlier::{estimate_absolute_pose, settings::MetasacSettings, types::DataMatrix};
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

#[derive(Clone, Debug)]
struct Observation {
    point_3d: Vector3<f64>,
    point_2d: (f64, f64),
    expected_inlier: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Absolute Pose Estimation Example ===\n");

    let n_inliers = 40;
    let n_outliers = 10;
    let mut rng = StdRng::seed_from_u64(0xA85E_FACE);

    let rotation_gt = UnitQuaternion::from_euler_angles(0.12, -0.18, 0.05);
    let translation_gt = Translation3::new(0.18, -0.12, 0.45);

    let mut observations = Vec::with_capacity(n_inliers + n_outliers);
    for _ in 0..n_inliers {
        let world = Vector3::new(
            rng.random_range(-1.5..1.5),
            rng.random_range(-1.2..1.2),
            rng.random_range(4.0..7.5),
        );
        let image = project(rotation_gt.transform_vector(&world) + translation_gt.vector);
        observations.push(Observation {
            point_3d: world,
            point_2d: (
                image.0 + rng.random_range(-0.0015..0.0015),
                image.1 + rng.random_range(-0.0015..0.0015),
            ),
            expected_inlier: true,
        });
    }

    for _ in 0..n_outliers {
        observations.push(Observation {
            point_3d: Vector3::new(
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(3.5..8.0),
            ),
            point_2d: (rng.random_range(-0.6..0.6), rng.random_range(-0.5..0.5)),
            expected_inlier: false,
        });
    }

    observations.shuffle(&mut rng);

    let mut points_3d = DataMatrix::zeros(observations.len(), 3);
    let mut points_2d = DataMatrix::zeros(observations.len(), 2);
    for (idx, obs) in observations.iter().enumerate() {
        points_3d.set(idx, 0, obs.point_3d.x);
        points_3d.set(idx, 1, obs.point_3d.y);
        points_3d.set(idx, 2, obs.point_3d.z);
        points_2d.set(idx, 0, obs.point_2d.0);
        points_2d.set(idx, 1, obs.point_2d.1);
    }

    let settings = MetasacSettings {
        min_iterations: 500,
        max_iterations: 5_000,
        confidence: 0.999,
        ..MetasacSettings::default()
    };
    let threshold = 0.01;
    let result = estimate_absolute_pose(&points_3d, &points_2d, threshold, Some(settings))?;

    let recovered_true_inliers = result
        .inliers
        .iter()
        .filter(|&&idx| observations[idx].expected_inlier)
        .count();
    let translation_error = (result.model.translation.vector - translation_gt.vector).norm();
    let rotation_error_deg = (result.model.rotation.inverse() * rotation_gt)
        .angle()
        .to_degrees();

    println!(
        "Recovered {} inliers out of {} correspondences",
        result.inliers.len(),
        observations.len()
    );
    println!("Correctly kept {recovered_true_inliers} / {n_inliers} true inliers");
    println!("Iterations: {}", result.iterations);
    println!("Score: {:?}", result.score);

    println!(
        "\nEstimated translation: {:?}",
        result.model.translation.vector
    );
    println!("Ground-truth translation: {:?}", translation_gt.vector);
    println!("Translation error norm: {translation_error:.6}");

    println!("\nEstimated rotation matrix:");
    let rotation_matrix = result.model.rotation.to_rotation_matrix();
    for row in 0..3 {
        println!(
            "  [{:9.5}, {:9.5}, {:9.5}]",
            rotation_matrix[(row, 0)],
            rotation_matrix[(row, 1)],
            rotation_matrix[(row, 2)]
        );
    }
    println!("Rotation error: {rotation_error_deg:.4} deg");

    Ok(())
}

fn project(point: Vector3<f64>) -> (f64, f64) {
    (point.x / point.z, point.y / point.z)
}
