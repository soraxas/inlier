use inlier::types::DataMatrix;
use nalgebra::{UnitQuaternion, Vector3};
use rand::Rng;
/// Generate a synthetic pair of point clouds with a linear scale drift along z.
///
/// Returns a DataMatrix with rows [sx, sy, sz, tx, ty, tz] for the surviving correspondences.
fn generate_linear_drift_dataset(
    num_points: usize,
    drop_src_frac: f64,
    drop_tgt_frac: f64,
    noise_sigma: f64,
    base_scale: f64,
    drift_factor: f64,
) -> DataMatrix {
    let mut rng = rand::rng();

    let rot = UnitQuaternion::from_euler_angles(0.1, -0.05, 0.07);
    let trans = Vector3::new(0.2, -0.1, 0.3);

    let mut src: Vec<Vector3<f64>> = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        src.push(Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ));
    }

    let mut rows = Vec::new();
    for p in src {
        if rng.random_range(0.0..1.0) < drop_src_frac {
            continue;
        }
        let z = p.z;
        let local_scale = base_scale * (1.0 + drift_factor * z);

        // Transform with rotation, translation, and local scale on the target.
        let p_t = local_scale * (rot.transform_vector(&p)) + trans;

        // Drop some target points independently.
        if rng.random_range(0.0..1.0) < drop_tgt_frac {
            continue;
        }

        let noise_src = Vector3::new(
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
        );
        let noise_tgt = Vector3::new(
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
        );

        let ps = p + noise_src;
        let pt = p_t + noise_tgt;

        rows.extend_from_slice(&[ps.x, ps.y, ps.z, pt.x, pt.y, pt.z]);
    }

    let n = rows.len() / 6;
    inlier::types::DataMatrix::from_row_slice(n, 6, &rows)
}

/// Generate a synthetic pair with a sinusoidal scale drift along z.
fn generate_nonlinear_drift_dataset(
    num_points: usize,
    drop_src_frac: f64,
    drop_tgt_frac: f64,
    noise_sigma: f64,
    base_scale: f64,
    drift_factor: f64,
) -> DataMatrix {
    let mut rng = rand::rng();

    let rot = UnitQuaternion::from_euler_angles(-0.2, 0.05, 0.1);
    let trans = Vector3::new(-0.3, 0.4, -0.1);

    let mut src: Vec<Vector3<f64>> = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        src.push(Vector3::new(
            rng.random_range(-1.5..1.5),
            rng.random_range(-1.5..1.5),
            rng.random_range(-1.5..1.5),
        ));
    }

    let mut rows = Vec::new();
    for p in src {
        if rng.random_range(0.0..1.0) < drop_src_frac {
            continue;
        }
        let z = p.z;
        let local_scale =
            base_scale * (1.0 + drift_factor * (2.0 * std::f64::consts::PI * z).sin());

        let p_t = local_scale * (rot.transform_vector(&p)) + trans;

        if rng.random_range(0.0..1.0) < drop_tgt_frac {
            continue;
        }

        let noise_src = Vector3::new(
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
        );
        let noise_tgt = Vector3::new(
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
            rng.random_range(-noise_sigma..noise_sigma),
        );

        let ps = p + noise_src;
        let pt = p_t + noise_tgt;
        rows.extend_from_slice(&[ps.x, ps.y, ps.z, pt.x, pt.y, pt.z]);
    }

    let n = rows.len() / 6;
    inlier::types::DataMatrix::from_row_slice(n, 6, &rows)
}

#[test]
fn linear_drift_dataset_has_points() {
    let data = generate_linear_drift_dataset(200, 0.2, 0.2, 0.01, 1.2, 0.1);
    assert!(data.n_points() >= 3);
}

#[test]
fn nonlinear_drift_dataset_has_points() {
    let data = generate_nonlinear_drift_dataset(200, 0.2, 0.2, 0.01, 1.1, 0.2);
    assert!(data.n_points() >= 3);
}
