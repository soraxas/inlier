use inlier::estimators::{
    LineEstimator, PlaneEstimator, RigidTransformEstimator, SimilarityTransformEstimator,
};
use inlier::models::Plane3;
use inlier::types::DataMatrix;
use inlier::{Estimator, MetasacSettings, estimate_line, estimate_plane, estimate_rigid_transform};
use nalgebra::{Point3, Vector3};

#[derive(Clone)]
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / ((1_u64 << 31) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn ransac_settings(seed: u64) -> MetasacSettings {
    MetasacSettings {
        min_iterations: 250,
        max_iterations: 250,
        max_sampling_attempts: 100,
        rng_seed: Some(seed),
        ..Default::default()
    }
}

fn plane_basis(normal: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let axis = if normal.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let u = normal.cross(&axis).normalize();
    let v = normal.cross(&u).normalize();
    (u, v)
}

fn deterministic_plane_cloud(scale: f64) -> (DataMatrix, Vector3<f64>, f64) {
    let normal = Vector3::new(0.21, -0.43, 0.88).normalize();
    let d = -1.7 * scale;
    let (u, v) = plane_basis(normal);
    let mut rows = Vec::new();

    for i in 0..90 {
        let a = ((i % 10) as f64 - 4.5) * 0.42 * scale;
        let b = ((i / 10) as f64 - 4.0) * 0.37 * scale;
        let base = a * u + b * v;
        let on_plane = base - normal * (normal.dot(&base) + d);
        let signed_noise = ((i % 7) as f64 - 3.0) * 1.5e-4 * scale;
        let p = on_plane + normal * signed_noise;
        rows.extend([p.x, p.y, p.z]);
    }

    let mut rng = Lcg(0xD1CE_BA5E);
    for _ in 0..18 {
        let p = Vector3::new(
            rng.range(-3.0, 3.0) * scale,
            rng.range(-3.0, 3.0) * scale,
            rng.range(-3.0, 3.0) * scale,
        ) + normal * (6.0 * scale);
        rows.extend([p.x, p.y, p.z]);
    }

    (
        DataMatrix::from_row_slice(rows.len() / 3, 3, &rows),
        normal,
        d,
    )
}

fn oriented_plane_parts(
    normal: Vector3<f64>,
    d: f64,
    expected_normal: Vector3<f64>,
) -> (Vector3<f64>, f64) {
    if normal.dot(&expected_normal) < 0.0 {
        (-normal, -d)
    } else {
        (normal, d)
    }
}

fn max_residual(points: &DataMatrix, normal: Vector3<f64>, d: f64, inliers: &[usize]) -> f64 {
    inliers
        .iter()
        .map(|&idx| {
            (normal.x * points.get(idx, 0)
                + normal.y * points.get(idx, 1)
                + normal.z * points.get(idx, 2)
                + d)
                .abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn plane3_normalization_handles_extreme_and_degenerate_coefficients() {
    let plane = Plane3::new(3.0e100, -4.0e100, 12.0e100, 5.0e100);
    assert!(plane.normal.iter().all(|v| v.is_finite()));
    assert!(plane.d.is_finite());
    assert!((plane.normal.norm() - 1.0).abs() < 1e-12);
    assert!((plane.normal.x - 3.0 / 13.0).abs() < 1e-12);
    assert!((plane.normal.y + 4.0 / 13.0).abs() < 1e-12);
    assert!((plane.normal.z - 12.0 / 13.0).abs() < 1e-12);
    assert!((plane.d - 5.0 / 13.0).abs() < 1e-12);

    let degenerate = Plane3::new(0.0, 0.0, 0.0, f64::NAN);
    assert_eq!(degenerate.normal, Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(degenerate.d, 0.0);
}

#[test]
fn estimate_plane_is_stable_across_coordinate_scales() {
    for scale in [1e-3, 1.0, 1e6] {
        let (points, expected_normal, expected_d) = deterministic_plane_cloud(scale);
        let threshold = 0.01 * scale;
        let result = estimate_plane(&points, threshold, Some(ransac_settings(0x5157_A11E)))
            .unwrap_or_else(|err| panic!("scale {scale} failed: {err}"));
        let (normal, d) =
            oriented_plane_parts(result.model.normal, result.model.d, expected_normal);

        assert!(
            normal.iter().all(|v| v.is_finite()) && d.is_finite(),
            "scale {scale}: model contains non-finite values"
        );
        assert!(
            normal.dot(&expected_normal) > 0.99999,
            "scale {scale}: normal {normal:?} expected {expected_normal:?}"
        );
        assert!(
            (d - expected_d).abs() <= 3.0 * threshold,
            "scale {scale}: d={d}, expected {expected_d}, threshold {threshold}"
        );
        assert!(
            result.inliers.len() >= 85,
            "scale {scale}: too few inliers: {}",
            result.inliers.len()
        );
        assert!(
            max_residual(&points, normal, d, &result.inliers) <= threshold * 1.2,
            "scale {scale}: inlier residual exceeded threshold"
        );
    }
}

#[test]
fn plane_estimator_rejects_nearly_collinear_minimal_samples() {
    let data = DataMatrix::from_row_slice(3, 3, &[0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 2e-6, 1e-18, 0.0]);
    let estimator = PlaneEstimator::new();
    assert!(!estimator.is_valid_sample(&data, &[0, 1, 2]));
    assert!(estimator.estimate_model(&data, &[0, 1, 2]).is_empty());
}

#[derive(Debug)]
#[allow(dead_code)]
struct PlaneFitSnapshot {
    inliers: usize,
    iterations: usize,
    normal_ppm: [i64; 3],
    d_ppm: i64,
    max_inlier_residual_ppm: i64,
}

#[test]
fn deterministic_plane_fit_snapshot() {
    let (points, expected_normal, _) = deterministic_plane_cloud(1.0);
    let result = estimate_plane(&points, 0.01, Some(ransac_settings(0x5157_A11E))).unwrap();
    let (normal, d) = oriented_plane_parts(result.model.normal, result.model.d, expected_normal);
    let summary = PlaneFitSnapshot {
        inliers: result.inliers.len(),
        iterations: result.iterations,
        normal_ppm: [
            (normal.x * 1_000_000.0).round() as i64,
            (normal.y * 1_000_000.0).round() as i64,
            (normal.z * 1_000_000.0).round() as i64,
        ],
        d_ppm: (d * 1_000_000.0).round() as i64,
        max_inlier_residual_ppm: (max_residual(&points, normal, d, &result.inliers) * 1_000_000.0)
            .round() as i64,
    };

    insta::assert_debug_snapshot!(summary, @r###"
PlaneFitSnapshot {
    inliers: 90,
    iterations: 250,
    normal_ppm: [
        209446,
        -429610,
        878389,
    ],
    d_ppm: -1699925,
    max_inlier_residual_ppm: 1050,
}
"###);
}

#[test]
fn estimate_plane_reports_degenerate_inputs_without_panic() {
    let too_few = DataMatrix::from_row_slice(2, 3, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    assert!(estimate_plane(&too_few, 0.1, Some(ransac_settings(1))).is_err());

    let wrong_dims = DataMatrix::zeros(8, 2);
    assert!(estimate_plane(&wrong_dims, 0.1, Some(ransac_settings(1))).is_err());

    let identical = DataMatrix::from_row_slice(
        8,
        3,
        &[
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0,
            3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
        ],
    );
    assert!(estimate_plane(&identical, 0.1, Some(ransac_settings(1))).is_err());

    let with_nan = DataMatrix::from_row_slice(
        4,
        3,
        &[
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            f64::NAN,
            0.0,
            0.0,
            1.0,
        ],
    );
    assert!(estimate_plane(&with_nan, 0.1, Some(ransac_settings(1))).is_err());
}

#[test]
fn plane_estimator_zero_weights_can_mask_outliers() {
    let data = DataMatrix::from_row_slice(
        6,
        3,
        &[
            -1.0, -1.0, 2.0, 1.0, -1.0, 2.0, -1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 30.0, -40.0, 100.0,
            -20.0, 60.0, -80.0,
        ],
    );
    let weights = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let sample: Vec<usize> = (0..6).collect();
    let models = PlaneEstimator::new().estimate_model_nonminimal(&data, &sample, Some(&weights));
    assert_eq!(models.len(), 1);

    let plane = &models[0];
    assert!(
        plane.normal.z.abs() > 0.999999,
        "weighted fit should recover z-plane: {:?}",
        plane.normal
    );
    assert!(
        plane.distance(0.0, 0.0, 2.0) < 1e-9,
        "weighted fit should pass through z=2"
    );
}

#[test]
fn plane_estimator_rejects_invalid_nonminimal_weights() {
    let data = DataMatrix::from_row_slice(
        4,
        3,
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    );
    let sample: Vec<usize> = (0..4).collect();
    let estimator = PlaneEstimator::new();

    for weights in [
        vec![1.0, -1.0, 1.0, 1.0],
        vec![1.0, f64::NAN, 1.0, 1.0],
        vec![1.0, f64::INFINITY, 1.0, 1.0],
        vec![1.0, 1.0],
    ] {
        let models = estimator.estimate_model_nonminimal(&data, &sample, Some(&weights));
        assert!(
            models.is_empty(),
            "invalid weights should produce no model: {weights:?}"
        );
    }
}

#[test]
fn estimate_line_reports_collapsed_inputs_without_panic() {
    let points = DataMatrix::from_row_slice(
        8,
        2,
        &[
            4.0, -2.0, 4.0, -2.0, 4.0, -2.0, 4.0, -2.0, 4.0, -2.0, 4.0, -2.0, 4.0, -2.0, 4.0, -2.0,
        ],
    );
    assert!(estimate_line(&points, 0.1, Some(ransac_settings(7))).is_err());
}

#[test]
fn estimate_line_rejects_non_finite_inputs() {
    let points =
        DataMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 1.0, f64::INFINITY, 2.0, 3.0, 3.0]);
    assert!(estimate_line(&points, 0.1, Some(ransac_settings(7))).is_err());
}

#[test]
fn line_estimator_minimal_fit_handles_large_offsets() {
    let x0 = 1.0e12;
    let x1 = x0 + 2048.0;
    let y0 = 3.0 * x0 + 7.0;
    let y1 = 3.0 * x1 + 7.0;
    let data = DataMatrix::from_row_slice(2, 2, &[x0, y0, x1, y1]);

    let models = LineEstimator::new().estimate_model(&data, &[0, 1]);
    assert_eq!(models.len(), 1);
    let line = &models[0];
    assert!(
        line.params.iter().all(|v| v.is_finite()),
        "line contains non-finite coefficients: {:?}",
        line.params
    );
    assert!(
        (line.params[0].hypot(line.params[1]) - 1.0).abs() < 1e-12,
        "line normal should stay unit length"
    );
    assert!(
        (line.params[0].abs() - (3.0_f64 / 10.0_f64.sqrt())).abs() < 1e-12,
        "line normal should encode slope 3: {:?}",
        line.params
    );
    assert!(
        (line.params[1].abs() - (1.0_f64 / 10.0_f64.sqrt())).abs() < 1e-12,
        "line normal should encode slope 3: {:?}",
        line.params
    );
}

#[test]
fn rigid_and_similarity_estimators_reject_collapsed_correspondences() {
    let data = DataMatrix::from_row_slice(
        3,
        6,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            6.0,
        ],
    );
    let sample = [0, 1, 2];
    assert!(
        RigidTransformEstimator::new()
            .estimate_model(&data, &sample)
            .is_empty()
    );
    assert!(
        SimilarityTransformEstimator::new()
            .estimate_model(&data, &sample)
            .is_empty()
    );
}

#[test]
fn estimate_rigid_transform_rejects_degenerate_public_inputs() {
    let two_src = DataMatrix::from_row_slice(2, 3, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let two_dst = DataMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 2.0, 2.0, 3.0]);
    assert!(estimate_rigid_transform(&two_src, &two_dst, 0.1, Some(ransac_settings(8))).is_err());

    let src = DataMatrix::from_row_slice(3, 3, &[0.0, 0.0, 0.0, 1.0, f64::NAN, 0.0, 0.0, 1.0, 0.0]);
    let dst = DataMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 3.0, 3.0]);
    assert!(estimate_rigid_transform(&src, &dst, 0.1, Some(ransac_settings(8))).is_err());
}

#[test]
fn rigid_transform_recovers_small_shape_at_large_world_offset() {
    let theta = 0.37_f64;
    let c = theta.cos();
    let s = theta.sin();
    let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let translation = Vector3::new(-4.0e8, 2.5e8, -7.0e8);
    let base = Vector3::new(1.0e9, -2.0e9, 3.0e9);
    let offsets = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(3.0, 0.0, 0.0),
        Vector3::new(0.0, 4.0, 0.0),
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(2.0, 1.0, 3.0),
    ];
    let mut rows = Vec::new();
    let mut expected = Vec::new();
    for p in offsets.into_iter().map(|offset| base + offset) {
        let q = Vector3::new(
            rotation[0][0] * p.x + rotation[0][1] * p.y + rotation[0][2] * p.z,
            rotation[1][0] * p.x + rotation[1][1] * p.y + rotation[1][2] * p.z,
            rotation[2][0] * p.x + rotation[2][1] * p.y + rotation[2][2] * p.z,
        ) + translation;
        rows.extend([p.x, p.y, p.z, q.x, q.y, q.z]);
        expected.push((p, q));
    }

    let data = DataMatrix::from_row_slice(expected.len(), 6, &rows);
    let sample: Vec<usize> = (0..expected.len()).collect();
    let models = RigidTransformEstimator::new().estimate_model(&data, &sample);
    assert_eq!(models.len(), 1);
    let model = &models[0];
    let det = model.rotation.to_rotation_matrix().matrix().determinant();
    assert!((det - 1.0).abs() < 1e-10, "rotation determinant: {det}");

    for (p, q) in expected {
        let predicted =
            model.rotation.transform_point(&Point3::from(p)).coords + model.translation.vector;
        assert!(
            (predicted - q).norm() < 1e-4,
            "large-offset transform residual too high: {}",
            (predicted - q).norm()
        );
    }
}
