//! Example: Homography estimation from real-image correspondences
//!
//! This example demonstrates homography estimation using landmark correspondences
//! derived from a real image pair. The companion Python visualizer fetches the
//! backing images from `inlier-data` via `pooch`.
//! It writes a small result report that can be visualized with
//! `python/demo_homography_scene.py`.

use inlier::{estimate_homography, types::DataMatrix};
use nalgebra::{Matrix3, Vector3};
use std::{
    collections::HashSet,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
struct MatchRecord {
    label: String,
    src: (f64, f64),
    dst: (f64, f64),
    expected_inlier: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let asset_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/homography");
    let match_file = asset_dir.join("correspondences.tsv");
    let matrix_output = asset_dir.join("estimated_homography_matrix.txt");
    let report_output = asset_dir.join("estimated_match_report.tsv");

    if !match_file.exists() {
        return Err(format!("required asset not found: {}", match_file.display()).into());
    }

    let matches = load_matches(&match_file)?;
    let mut points1 = DataMatrix::zeros(matches.len(), 2);
    let mut points2 = DataMatrix::zeros(matches.len(), 2);

    for (idx, record) in matches.iter().enumerate() {
        points1.set(idx, 0, record.src.0);
        points1.set(idx, 1, record.src.1);
        points2.set(idx, 0, record.dst.0);
        points2.set(idx, 1, record.dst.1);
    }

    println!("=== Homography Estimation with Example Images ===\n");
    println!("Real-image pair: fetched by python/demo_homography_scene.py from inlier-data");
    println!("Correspondence file: {}", match_file.display());
    println!("Loaded {} tentative landmark matches", matches.len());

    let expected_inliers = matches.iter().filter(|m| m.expected_inlier).count();
    let expected_outliers = matches.len() - expected_inliers;
    println!("Expected curated inliers: {expected_inliers}");
    println!("Expected curated outliers: {expected_outliers}\n");

    let threshold = 6.0;
    let result = estimate_homography(&points1, &points2, threshold, None)?;
    let reported_inliers = result.inliers.len();
    let inlier_set = recover_inliers(&result.model.h, &matches, threshold);

    println!("Estimation results:");
    println!(
        "  Recovered {} inliers out of {} matches",
        inlier_set.len(),
        matches.len()
    );
    println!("  API-reported inliers: {reported_inliers}");
    println!("  Score: {:?}", result.score);
    println!("  Iterations: {}", result.iterations);

    println!("\nEstimated homography matrix:");
    for row in 0..3 {
        println!(
            "  [{:8.4}, {:8.4}, {:8.4}]",
            result.model.h[(row, 0)],
            result.model.h[(row, 1)],
            result.model.h[(row, 2)]
        );
    }

    let correctly_classified = matches
        .iter()
        .enumerate()
        .filter(|(idx, record)| inlier_set.contains(idx) == record.expected_inlier)
        .count();
    println!(
        "\nCorrectly classified {correctly_classified} / {} curated matches",
        matches.len()
    );

    let mut total_inlier_error = 0.0;
    let mut inlier_error_count = 0usize;
    let mut report = String::from(
        "# label\tsrc_x\tsrc_y\tdst_x\tdst_y\texpected_inlier\testimated_inlier\treprojection_error\n",
    );
    for (idx, record) in matches.iter().enumerate() {
        let projected = project(&result.model.h, record.src);
        let error = distance(projected, record.dst);
        let estimated_inlier = inlier_set.contains(&idx);
        let verdict = if estimated_inlier {
            "INLIER"
        } else {
            "OUTLIER"
        };
        println!(
            "  {label:>2}: projected=({px:6.2}, {py:6.2}) target=({tx:6.2}, {ty:6.2}) error={err:5.2} px [{verdict}]",
            label = record.label,
            px = projected.0,
            py = projected.1,
            tx = record.dst.0,
            ty = record.dst.1,
            err = error,
        );
        report.push_str(&format!(
            "{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}\n",
            record.label,
            record.src.0,
            record.src.1,
            record.dst.0,
            record.dst.1,
            record.expected_inlier,
            estimated_inlier,
            error
        ));
        if estimated_inlier {
            total_inlier_error += error;
            inlier_error_count += 1;
        }
    }
    if inlier_error_count > 0 {
        println!(
            "\nAverage reprojection error over estimated inliers: {:.3} px",
            total_inlier_error / inlier_error_count as f64
        );
    }

    write_matrix(&matrix_output, &result.model.h)?;
    fs::write(&report_output, report)?;

    println!("\nMatrix written to: {}", matrix_output.display());
    println!("Match report written to: {}", report_output.display());
    println!(
        "Render the scene with: python python/demo_homography_scene.py --asset-dir {}",
        asset_dir.display()
    );

    Ok(())
}

fn load_matches(path: &Path) -> Result<Vec<MatchRecord>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut matches = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<_> = trimmed.split('\t').collect();
        if parts.len() != 6 {
            return Err(format!("invalid correspondence line: {trimmed}").into());
        }

        matches.push(MatchRecord {
            label: parts[0].to_string(),
            src: (parts[1].parse()?, parts[2].parse()?),
            dst: (parts[3].parse()?, parts[4].parse()?),
            expected_inlier: parts[5].parse()?,
        });
    }

    if matches.len() < 4 {
        return Err("need at least four correspondences for homography estimation".into());
    }

    Ok(matches)
}

fn write_matrix(path: &PathBuf, matrix: &Matrix3<f64>) -> Result<(), Box<dyn Error>> {
    let content = format!(
        "{:.10} {:.10} {:.10}\n{:.10} {:.10} {:.10}\n{:.10} {:.10} {:.10}\n",
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
    );
    fs::write(path, content)?;
    Ok(())
}

fn recover_inliers(
    homography: &Matrix3<f64>,
    matches: &[MatchRecord],
    threshold: f64,
) -> HashSet<usize> {
    matches
        .iter()
        .enumerate()
        .filter_map(|(idx, record)| {
            let error = distance(project(homography, record.src), record.dst);
            if error <= threshold { Some(idx) } else { None }
        })
        .collect()
}

fn project(h: &Matrix3<f64>, point: (f64, f64)) -> (f64, f64) {
    let homogeneous = h * Vector3::new(point.0, point.1, 1.0);
    (homogeneous.x / homogeneous.z, homogeneous.y / homogeneous.z)
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}
