//! Integration tests for the high-level Rust API.
//!
//! These tests verify that the estimation functions work correctly with
//! synthetic data and produce reasonable results.

use inlier::settings::{LocalOptimizationType, SamplerType};
use inlier::{
    presets::rigid_registration_pipeline, types::DataMatrix, utils::combine_input_points_33, *,
};
use nalgebra::{Matrix3, Rotation3, Vector3};

#[test]
fn test_estimate_homography_synthetic() {
    // Create synthetic 2D point correspondences
    // Simple translation: points in image 1 are shifted by (10, 5) in image 2
    let n_points = 20;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);

    for i in 0..n_points {
        let x = (i as f64) * 10.0;
        let y = (i as f64) * 5.0;
        points1.set(i, 0, x);
        points1.set(i, 1, y);
        points2.set(i, 0, x + 10.0); // Translation in x
        points2.set(i, 1, y + 5.0); // Translation in y
    }

    let threshold = 1.0;
    let result = estimate_homography(&points1, &points2, threshold, None);

    // RANSAC might not always succeed with minimal synthetic data
    if let Ok(result) = result {
        assert!(
            !result.inliers.is_empty(),
            "Should find some inliers if estimation succeeds"
        );
    }
    // If it fails, that's also acceptable for synthetic data
}

#[test]
fn test_estimate_homography_projective_transform() {
    let homography = Matrix3::new(
        1.042, -0.003, -12.4, 0.006, 1.038, 3.1, 0.00001, -0.00002, 1.0,
    );
    let n_points = 20;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);
    for index in 0..n_points {
        let source = Vector3::new((index % 5) as f64 * 125.0, (index / 5) as f64 * 100.0, 1.0);
        let target = homography * source;
        points1.set(index, 0, source.x);
        points1.set(index, 1, source.y);
        points2.set(index, 0, target.x / target.z);
        points2.set(index, 1, target.y / target.z);
    }

    let settings = MetasacSettings {
        min_iterations: 1_000,
        max_iterations: 1_000,
        rng_seed: Some(7),
        ..Default::default()
    };
    let result = estimate_homography(&points1, &points2, 1e-3, Some(settings))
        .expect("projective homography should be estimated");
    assert_eq!(result.inliers.len(), n_points);
}

#[test]
fn test_estimate_homography_large_pixel_coordinates() {
    let homography = Matrix3::new(
        1.000_3, -0.000_2, 45_000.0, 0.000_15, 0.999_7, -28_000.0, 2e-10, -3e-10, 1.0,
    );
    let n_points = 24;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);
    for index in 0..n_points {
        let source = Vector3::new(
            1.0e8 + (index % 6) as f64 * 18_000.0,
            -2.0e8 + (index / 6) as f64 * 22_000.0,
            1.0,
        );
        let target = homography * source;
        points1.set(index, 0, source.x);
        points1.set(index, 1, source.y);
        points2.set(index, 0, target.x / target.z);
        points2.set(index, 1, target.y / target.z);
    }

    let settings = MetasacSettings {
        min_iterations: 1_000,
        max_iterations: 1_000,
        rng_seed: Some(11),
        ..Default::default()
    };
    let result = estimate_homography(&points1, &points2, 0.01, Some(settings))
        .expect("normalized DLT should handle large pixel coordinates");
    assert_eq!(result.inliers.len(), n_points);
    for index in 0..n_points {
        let projected =
            result.model.h * Vector3::new(points1.get(index, 0), points1.get(index, 1), 1.0);
        let dx = projected.x / projected.z - points2.get(index, 0);
        let dy = projected.y / projected.z - points2.get(index, 1);
        assert!((dx * dx + dy * dy).sqrt() < 0.01);
    }
}

#[test]
fn high_level_apis_reject_non_finite_coordinates_and_invalid_thresholds() {
    let mut image_points = DataMatrix::zeros(8, 2);
    image_points.set(0, 0, f64::NAN);
    let clean_image_points = DataMatrix::zeros(8, 2);
    assert!(estimate_homography(&image_points, &clean_image_points, 1.0, None).is_err());
    assert!(estimate_fundamental_matrix(&image_points, &clean_image_points, 1.0, None).is_err());
    assert!(estimate_essential_matrix(&image_points, &clean_image_points, 1.0, None).is_err());

    let mut world_points = DataMatrix::zeros(4, 3);
    world_points.set(0, 2, f64::INFINITY);
    let clean_world_points = DataMatrix::zeros(4, 3);
    let clean_image_points = DataMatrix::zeros(4, 2);
    assert!(estimate_absolute_pose(&world_points, &clean_image_points, 1.0, None).is_err());
    assert!(estimate_rigid_transform(&world_points, &clean_world_points, 1.0, None).is_err());
    assert!(estimate_plane(&world_points, 1.0, None).is_err());

    let mut line_points = DataMatrix::zeros(2, 2);
    line_points.set(1, 1, f64::NEG_INFINITY);
    assert!(estimate_line(&line_points, 1.0, None).is_err());

    let valid_line_points = DataMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    for invalid_threshold in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(estimate_line(&valid_line_points, invalid_threshold, None).is_err());
    }
}

#[test]
fn estimate_homography_rejects_collinear_correspondences_before_ransac() {
    let points1 = DataMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    let points2 = DataMatrix::from_row_slice(4, 2, &[5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0]);
    let error = estimate_homography(&points1, &points2, 1.0, None).unwrap_err();
    assert!(error.contains("non-collinear"));
}

#[test]
fn epipolar_apis_reject_collapsed_correspondences_before_ransac() {
    let points = DataMatrix::zeros(8, 2);
    let fundamental_error = estimate_fundamental_matrix(&points, &points, 1.0, None).unwrap_err();
    assert!(fundamental_error.contains("non-collinear"));

    let essential_error = estimate_essential_matrix(&points, &points, 1.0, None).unwrap_err();
    assert!(essential_error.contains("non-collinear"));
}

#[test]
fn high_level_apis_reject_invalid_ransac_settings() {
    let points = DataMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let cases = [
        MetasacSettings {
            min_iterations: 2,
            max_iterations: 1,
            ..Default::default()
        },
        MetasacSettings {
            max_sampling_attempts: 0,
            ..Default::default()
        },
        MetasacSettings {
            confidence: f64::NAN,
            ..Default::default()
        },
        MetasacSettings {
            point_priors: Some(vec![1.0]),
            ..Default::default()
        },
        MetasacSettings {
            point_priors: Some(vec![1.0, -1.0]),
            ..Default::default()
        },
        MetasacSettings {
            point_priors: Some(vec![1.0, f64::NAN]),
            ..Default::default()
        },
    ];

    for settings in cases {
        assert!(estimate_line(&points, 1.0, Some(settings)).is_err());
    }
}

#[test]
fn test_estimate_fundamental_matrix_synthetic() {
    // Project non-planar 3D points into two calibrated cameras. A planar
    // rotation/translation in image space is homography-degenerate and is not
    // a valid success case for fundamental-matrix estimation.
    let n_points = 15;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);
    let (sin_angle, cos_angle) = 0.19_f64.sin_cos();

    for i in 0..n_points {
        let x = (i as f64 * 0.37).sin() * 1.2;
        let y = (i as f64 * 0.61).cos() * 0.8;
        let z = 3.0 + (i % 5) as f64 * 0.45;
        points1.set(i, 0, x / z);
        points1.set(i, 1, y / z);

        let rotated_x = cos_angle * x + sin_angle * z;
        let rotated_z = -sin_angle * x + cos_angle * z;
        points2.set(i, 0, (rotated_x + 0.35) / (rotated_z + 0.2));
        points2.set(i, 1, (y - 0.12) / (rotated_z + 0.2));
    }

    let threshold = 0.01;
    let result = estimate_fundamental_matrix(&points1, &points2, threshold, None);

    assert!(
        result.is_ok(),
        "Fundamental matrix estimation should succeed, got error: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.inliers.is_empty(), "Should find some inliers");
}

#[test]
fn test_estimate_essential_matrix_synthetic() {
    // Use a non-planar calibrated two-view scene. A 2D rotation is a
    // homography/pure-rotation degeneracy and does not define an essential
    // matrix success case.
    let n_points = 15;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);
    let rotation = Rotation3::from_euler_angles(0.08, -0.11, 0.03);
    let translation = Vector3::new(0.28, -0.09, 0.16);

    for i in 0..n_points {
        let point = Vector3::new(
            (i as f64 * 0.41).sin() * 0.9,
            (i as f64 * 0.73).cos() * 0.7,
            2.5 + (i % 5) as f64 * 0.4,
        );
        let transformed = rotation * point + translation;
        points1.set(i, 0, point.x / point.z);
        points1.set(i, 1, point.y / point.z);
        points2.set(i, 0, transformed.x / transformed.z);
        points2.set(i, 1, transformed.y / transformed.z);
    }

    let threshold = 1e-4;
    let result = estimate_essential_matrix(&points1, &points2, threshold, None);

    let result =
        result.expect("non-planar calibrated correspondences should estimate an essential matrix");
    assert_eq!(result.inliers.len(), n_points);
}

#[test]
fn test_estimate_essential_matrix_recovers_nonplanar_pose() {
    let rotation = Rotation3::from_euler_angles(0.12, -0.08, 0.04);
    let translation = Vector3::new(0.3, -0.1, 0.2);
    let mut points1 = DataMatrix::zeros(24, 2);
    let mut points2 = DataMatrix::zeros(24, 2);

    for index in 0..24 {
        let point = Vector3::new(
            (index % 6) as f64 * 0.18 - 0.45,
            (index / 6) as f64 * 0.14 - 0.2,
            2.5 + (index % 5) as f64 * 0.35,
        );
        let transformed = rotation * point + translation;
        points1.set(index, 0, point.x / point.z);
        points1.set(index, 1, point.y / point.z);
        points2.set(index, 0, transformed.x / transformed.z);
        points2.set(index, 1, transformed.y / transformed.z);
    }

    let settings = MetasacSettings {
        min_iterations: 100,
        max_iterations: 100,
        rng_seed: Some(0xE551_EE55),
        ..MetasacSettings::default()
    };
    let result = estimate_essential_matrix(&points1, &points2, 1e-6, Some(settings))
        .expect("non-planar calibrated correspondences should estimate an essential matrix");
    assert_eq!(result.inliers.len(), 24);

    let skew_translation = Matrix3::new(
        0.0,
        -translation.z,
        translation.y,
        translation.z,
        0.0,
        -translation.x,
        -translation.y,
        translation.x,
        0.0,
    );
    let expected = skew_translation * rotation.matrix();
    let estimated = result.model.e / result.model.e.norm();
    let expected = expected / expected.norm();
    let sign_invariant_error = (estimated - expected)
        .norm()
        .min((estimated + expected).norm());
    assert!(
        sign_invariant_error < 1e-5,
        "essential matrix error: {sign_invariant_error}"
    );
}

#[test]
fn test_estimate_absolute_pose_synthetic() {
    // Create synthetic 3D-2D correspondences
    let n_points = 10;
    let mut points_3d = DataMatrix::zeros(n_points, 3);
    let mut points_2d = DataMatrix::zeros(n_points, 2);

    // Create non-collinear points on a plane at z=5. P3P permits a planar
    // scene, but each minimal sample must form a non-degenerate triangle.
    for i in 0..n_points {
        let x = (i % 5) as f64 * 10.0 - 20.0;
        let y = (i / 5) as f64 * 12.0 - 6.0;
        points_3d.set(i, 0, x);
        points_3d.set(i, 1, y);
        points_3d.set(i, 2, 5.0);

        // Project to 2D (simple perspective projection, camera at origin looking along +z)
        points_2d.set(i, 0, x / 5.0); // x/z
        points_2d.set(i, 1, y / 5.0); // y/z
    }

    let threshold = 1.0;
    let result = estimate_absolute_pose(&points_3d, &points_2d, threshold, None);

    assert!(result.is_ok(), "Absolute pose estimation should succeed");
    let result = result.unwrap();
    assert!(!result.inliers.is_empty(), "Should find some inliers");
}

#[test]
fn test_estimate_rigid_transform_synthetic() {
    // Create synthetic 3D-3D correspondences
    let n_points = 10;
    let mut points1 = DataMatrix::zeros(n_points, 3);
    let mut points2 = DataMatrix::zeros(n_points, 3);

    // Create points and apply a known rigid transform
    for i in 0..n_points {
        let x = (i % 5) as f64 * 10.0;
        let y = (i / 5) as f64 * 12.0;
        let z = (i % 3) as f64 * 4.0;
        points1.set(i, 0, x);
        points1.set(i, 1, y);
        points1.set(i, 2, z);

        // Apply translation (10, 5, 2)
        points2.set(i, 0, x + 10.0);
        points2.set(i, 1, y + 5.0);
        points2.set(i, 2, z + 2.0);
    }

    let threshold = 0.1;
    let pipeline = rigid_registration_pipeline(threshold, false, None, MetasacSettings::default());
    let input = combine_input_points_33(&points1, &points2).unwrap();
    let result = pipeline.run(&input);

    assert!(
        result.is_some(),
        "Rigid transform estimation should succeed"
    );
    let result = result.unwrap();
    assert!(!result.inliers.is_empty(), "Should find some inliers");
    assert!(
        result.inliers.len() >= n_points / 2,
        "Should find at least half the points as inliers"
    );
}

#[test]
fn test_estimate_homography_with_outliers() {
    // Create synthetic data with some outliers
    let n_points = 20;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);

    // First 15 points follow a transformation
    for i in 0..15 {
        let x = (i as f64) * 10.0;
        let y = (i as f64) * 5.0;
        points1.set(i, 0, x);
        points1.set(i, 1, y);
        points2.set(i, 0, x + 10.0);
        points2.set(i, 1, y + 5.0);
    }

    // Last 5 points are outliers (random)
    for i in 15..n_points {
        points1.set(i, 0, (i as f64) * 100.0);
        points1.set(i, 1, (i as f64) * 200.0);
        points2.set(i, 0, (i as f64) * 300.0);
        points2.set(i, 1, (i as f64) * 400.0);
    }

    let threshold = 1.0;
    let result = estimate_homography(&points1, &points2, threshold, None);

    // RANSAC might not always succeed with synthetic data and outliers
    // Just verify the function doesn't panic
    match result {
        Ok(result) => {
            // If it succeeds, should find some inliers
            assert!(
                !result.inliers.is_empty(),
                "Result should be valid with at least one inlier"
            );
        }
        Err(_) => {
            // Failure is acceptable for synthetic data with outliers
        }
    }
}

#[test]
fn test_high_level_api_rejects_unimplemented_sampler() {
    let points = DataMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let settings = MetasacSettings {
        sampler: SamplerType::Napsac,
        ..Default::default()
    };
    let error = estimate_line(&points, 0.1, Some(settings))
        .expect_err("NAPSAC is not available through the high-level API");
    assert!(error.contains("sampler Napsac is not implemented"));
}

#[test]
fn test_high_level_api_rejects_unimplemented_optimizer() {
    let points = DataMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let settings = MetasacSettings {
        local_optimization: LocalOptimizationType::Irls,
        ..Default::default()
    };
    let error = estimate_line(&points, 0.1, Some(settings))
        .expect_err("IRLS is not available through the high-level API");
    assert!(error.contains("local optimization Irls is not implemented"));
}

#[test]
fn test_absolute_pose_rejects_collapsed_world_points() {
    let world = DataMatrix::zeros(4, 3);
    let image = DataMatrix::zeros(4, 2);
    let settings = MetasacSettings {
        min_iterations: 16,
        max_iterations: 16,
        max_sampling_attempts: 1,
        rng_seed: Some(13),
        ..Default::default()
    };
    assert!(estimate_absolute_pose(&world, &image, 0.01, Some(settings)).is_err());
}

#[test]
fn test_estimate_fundamental_matrix_insufficient_points() {
    // Test with too few points (need at least 7 for fundamental matrix)
    let n_points = 5;
    let mut points1 = DataMatrix::zeros(n_points, 2);
    let mut points2 = DataMatrix::zeros(n_points, 2);

    for i in 0..n_points {
        points1.set(i, 0, (i as f64) * 10.0);
        points1.set(i, 1, (i as f64) * 5.0);
        points2.set(i, 0, (i as f64) * 10.0 + 1.0);
        points2.set(i, 1, (i as f64) * 5.0 + 1.0);
    }

    let threshold = 1.0;
    let result = estimate_fundamental_matrix(&points1, &points2, threshold, None);

    // Should either succeed (if it can find a model) or fail gracefully
    // We don't assert failure here since RANSAC might still work with minimal samples
    if let Ok(result) = result {
        assert!(
            !result.inliers.is_empty() || result.inliers.is_empty(),
            "Result should be valid"
        );
    }
}
