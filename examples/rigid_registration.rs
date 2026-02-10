//! Example: Similarity (scale + rotation + translation) registration using the TEASER-like preset.

use inlier::Pipeline;
use inlier::presets::similarity_registration::similarity_registration_pipeline;
use inlier::settings::MetasacSettings;
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();

    // Ground-truth transform.
    let scale_gt = 1.5;
    let rot_gt = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.5);
    let trans_gt = Translation3::new(0.2, -0.4, 0.6);

    // Generate random 3D points.
    let n = 100;
    let mut data = inlier::types::DataMatrix::zeros(n, 6);
    for i in 0..n {
        let p = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let p_rot = rot_gt.transform_point(&p.into());
        let p_t = scale_gt * p_rot.coords + trans_gt.vector;

        data.set(i, 0, p.x);
        data.set(i, 1, p.y);
        data.set(i, 2, p.z);

        // Add mild noise
        data.set(i, 3, p_t.x + rng.random_range(-0.01..0.01));
        data.set(i, 4, p_t.y + rng.random_range(-0.01..0.01));
        data.set(i, 5, p_t.z + rng.random_range(-0.01..0.01));
    }

    let settings = MetasacSettings {
        min_iterations: 500,
        max_iterations: 2000,
        confidence: 0.999,
        ..MetasacSettings::default()
    };

    let pipeline = similarity_registration_pipeline(0.05, false, None, settings);
    if let Some(result) = pipeline.run(&data) {
        println!("Estimated scale:    {:.4}", result.model.scale);
        println!(
            "Estimated rotation: {:.4?}",
            result.model.rotation.to_rotation_matrix()
        );
        println!(
            "Estimated translation: {:.4?}",
            result.model.translation.vector
        );
        println!("Inliers: {}", result.inliers.len());
        println!("Iterations: {}", result.iterations);
    } else {
        eprintln!("Registration failed");
    }

    Ok(())
}
