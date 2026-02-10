//! Test comparing with TEASER++ using proper TLS estimator

use inlier::presets::pointcloud_registration::estimate_scale_tls;
use inlier::types::DataMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_csv_points(path: &str) -> Result<DataMatrix, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let values: Result<Vec<f64>, _> =
            line.split(',').map(|s| s.trim().parse::<f64>()).collect();
        rows.push(values?);
    }

    if rows.is_empty() {
        return Err("No data in CSV".into());
    }

    let n_dims = rows.len();
    let n_points = rows[0].len();

    let mut flat = Vec::with_capacity(n_points * n_dims);
    for point_idx in 0..n_points {
        for dim_idx in 0..n_dims {
            flat.push(rows[dim_idx][point_idx]);
        }
    }

    Ok(DataMatrix::from_row_slice(n_points, n_dims, &flat))
}

#[test]
fn test_teaser_scale_tls() {
    let src_path = "TEASER-plusplus/test/teaser/data/registration_test/objectIn.csv";
    let dst_path = "TEASER-plusplus/test/teaser/data/registration_test/sceneIn.csv";

    if !std::path::Path::new(src_path).exists() {
        eprintln!("Skipping test: TEASER++ test data not found");
        return;
    }

    let src = load_csv_points(src_path).expect("Failed to load source points");
    let dst = load_csv_points(dst_path).expect("Failed to load target points");

    println!("Loaded {} source points", src.n_points());
    println!("Loaded {} target points", dst.n_points());

    // TEASER++ parameters from registration-test.cc
    let noise_bound = 0.0067364;
    let c_bar = 1.0;

    let result = estimate_scale_tls(&src, &dst, noise_bound, c_bar);

    assert!(result.is_some(), "Scale estimation should succeed");
    let (scale, inliers) = result.unwrap();

    let expected_scale = 0.955885;
    let n_inliers = inliers.iter().filter(|&&x| x).count();

    println!("\n=== TLS Scale Estimation ===");
    println!("Estimated scale: {}", scale);
    println!("Expected scale: {}", expected_scale);
    println!("Error: {:.6}", (scale - expected_scale).abs());
    println!(
        "Inliers: {} / {} ({:.1}%)",
        n_inliers,
        inliers.len(),
        100.0 * n_inliers as f64 / inliers.len() as f64
    );

    // TEASER++ uses 0.01 (1%) tolerance
    let tolerance = 0.01;
    assert!(
        (scale - expected_scale).abs() < tolerance,
        "Scale estimation: got {}, expected {}, diff {} (tolerance {})",
        scale,
        expected_scale,
        (scale - expected_scale).abs(),
        tolerance
    );
}
