//! Example: Rigid transform estimation from 3D-3D correspondences
//!
//! This example estimates a rotation + translation from synthetic point matches,
//! writes a small TSV report, and can be visualized with
//! `python/demo_rigid_transform_scene.py`.

use inlier::{api::estimate_rigid_transform, settings::MetasacSettings, types::DataMatrix};
use nalgebra::{Matrix4, Point3, Translation3, UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{collections::HashSet, error::Error, fs, path::Path};

#[derive(Clone, Debug)]
struct Correspondence {
    src: Vector3<f64>,
    dst: Vector3<f64>,
    expected_inlier: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Rigid Transform Estimation Example ===\n");

    let asset_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/rigid_transform");
    fs::create_dir_all(&asset_dir)?;
    let transform_output = asset_dir.join("estimated_rigid_transform.txt");
    let report_output = asset_dir.join("estimated_correspondence_report.tsv");

    let n_inliers = 48;
    let n_outliers = 12;
    let mut rng = StdRng::seed_from_u64(0xA11D_ED55);
    let rotation_gt = UnitQuaternion::from_euler_angles(0.25, -0.15, 0.35);
    let translation_gt = Translation3::new(0.65, -0.30, 0.45);

    let mut correspondences = Vec::with_capacity(n_inliers + n_outliers);
    for _ in 0..n_inliers {
        let src = Vector3::new(
            rng.random_range(-1.2..1.2),
            rng.random_range(-0.9..0.9),
            rng.random_range(-0.7..0.7),
        );
        let target_point = rotation_gt.transform_point(&Point3::from(src)).coords
            + translation_gt.vector
            + Vector3::new(
                rng.random_range(-0.01..0.01),
                rng.random_range(-0.01..0.01),
                rng.random_range(-0.01..0.01),
            );
        correspondences.push(Correspondence {
            src,
            dst: target_point,
            expected_inlier: true,
        });
    }

    for _ in 0..n_outliers {
        correspondences.push(Correspondence {
            src: Vector3::new(
                rng.random_range(-1.2..1.2),
                rng.random_range(-0.9..0.9),
                rng.random_range(-0.7..0.7),
            ),
            dst: Vector3::new(
                rng.random_range(-1.5..1.8),
                rng.random_range(-1.5..1.8),
                rng.random_range(-1.5..1.8),
            ),
            expected_inlier: false,
        });
    }

    correspondences.shuffle(&mut rng);

    let mut points_src = DataMatrix::zeros(correspondences.len(), 3);
    let mut points_tgt = DataMatrix::zeros(correspondences.len(), 3);
    for (idx, corr) in correspondences.iter().enumerate() {
        points_src.set(idx, 0, corr.src.x);
        points_src.set(idx, 1, corr.src.y);
        points_src.set(idx, 2, corr.src.z);
        points_tgt.set(idx, 0, corr.dst.x);
        points_tgt.set(idx, 1, corr.dst.y);
        points_tgt.set(idx, 2, corr.dst.z);
    }

    let settings = MetasacSettings {
        min_iterations: 400,
        max_iterations: 4_000,
        confidence: 0.999,
        ..MetasacSettings::default()
    };
    let threshold = 0.05;
    let result = estimate_rigid_transform(&points_src, &points_tgt, threshold, Some(settings))?;
    let inlier_set: HashSet<usize> = result.inliers.iter().copied().collect();

    let recovered_true_inliers = result
        .inliers
        .iter()
        .filter(|&&idx| correspondences[idx].expected_inlier)
        .count();
    let translation_error = (result.model.translation.vector - translation_gt.vector).norm();
    let rotation_error_deg = (result.model.rotation.inverse() * rotation_gt)
        .angle()
        .to_degrees();

    println!(
        "Recovered {} inliers out of {} correspondences",
        result.inliers.len(),
        correspondences.len()
    );
    println!("Correctly kept {recovered_true_inliers} / {n_inliers} true inliers");
    println!("Iterations: {}", result.iterations);
    println!("Score: {:?}", result.score);
    println!("Translation error norm: {translation_error:.6}");
    println!("Rotation error: {rotation_error_deg:.4} deg");

    let transform_matrix = result.model.to_matrix4();
    println!("\nEstimated rigid transform matrix:");
    for row in 0..4 {
        println!(
            "  [{:9.5}, {:9.5}, {:9.5}, {:9.5}]",
            transform_matrix[(row, 0)],
            transform_matrix[(row, 1)],
            transform_matrix[(row, 2)],
            transform_matrix[(row, 3)]
        );
    }

    let mut report = String::from(
        "# src_x\tsrc_y\tsrc_z\tdst_x\tdst_y\tdst_z\tpred_x\tpred_y\tpred_z\texpected_inlier\testimated_inlier\tresidual\n",
    );
    for (idx, corr) in correspondences.iter().enumerate() {
        let predicted = result
            .model
            .rotation
            .transform_point(&Point3::from(corr.src))
            .coords
            + result.model.translation.vector;
        let estimated_inlier = inlier_set.contains(&idx);
        let residual = (predicted - corr.dst).norm();
        report.push_str(&format!(
            "{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.6}\n",
            corr.src.x,
            corr.src.y,
            corr.src.z,
            corr.dst.x,
            corr.dst.y,
            corr.dst.z,
            predicted.x,
            predicted.y,
            predicted.z,
            corr.expected_inlier,
            estimated_inlier,
            residual
        ));
    }

    write_matrix4(&transform_output, &transform_matrix)?;
    fs::write(&report_output, report)?;

    println!("\nTransform written to: {}", transform_output.display());
    println!(
        "Correspondence report written to: {}",
        report_output.display()
    );
    println!(
        "Render the alignment with: python python/demo_rigid_transform_scene.py --asset-dir {}",
        asset_dir.display()
    );

    Ok(())
}

fn write_matrix4(path: &Path, matrix: &Matrix4<f64>) -> Result<(), Box<dyn Error>> {
    let content = format!(
        "{:.10} {:.10} {:.10} {:.10}\n{:.10} {:.10} {:.10} {:.10}\n{:.10} {:.10} {:.10} {:.10}\n{:.10} {:.10} {:.10} {:.10}\n",
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
        matrix[(0, 3)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(1, 3)],
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
        matrix[(2, 3)],
        matrix[(3, 0)],
        matrix[(3, 1)],
        matrix[(3, 2)],
        matrix[(3, 3)],
    );
    fs::write(path, content)?;
    Ok(())
}
