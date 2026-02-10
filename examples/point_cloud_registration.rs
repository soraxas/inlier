//! Example: TEASER-inspired point cloud registration (scale + rotation + translation).

use inlier::presets::pointcloud_registration::teaser_pointcloud_registration_pipeline;
use inlier::{Pipeline, settings::MetasacSettings};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load a simple XYZ point cloud (one point per line: x y z).
fn load_xyz(path: &str) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pts = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<_> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if parts.len() >= 3 {
            pts.push([parts[0], parts[1], parts[2]]);
        }
    }
    Ok(pts)
}

/// Simple mutual nearest-neighbor correspondences (brute force), capped at `max_corr`.
fn mutual_nn_pairs(src: &[[f64; 3]], tgt: &[[f64; 3]], max_corr: usize) -> Vec<(usize, usize)> {
    let mut src_to_tgt = vec![0usize; src.len()];
    let mut tgt_to_src = vec![0usize; tgt.len()];

    for (i, s) in src.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d2 = f64::MAX;
        for (j, t) in tgt.iter().enumerate() {
            let d2 = (s[0] - t[0]).powi(2) + (s[1] - t[1]).powi(2) + (s[2] - t[2]).powi(2);
            if d2 < best_d2 {
                best_d2 = d2;
                best = j;
            }
        }
        src_to_tgt[i] = best;
    }

    for (j, t) in tgt.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d2 = f64::MAX;
        for (i, s) in src.iter().enumerate() {
            let d2 = (s[0] - t[0]).powi(2) + (s[1] - t[1]).powi(2) + (s[2] - t[2]).powi(2);
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        tgt_to_src[j] = best;
    }

    let mut pairs = Vec::new();
    for (i, &j) in src_to_tgt.iter().enumerate() {
        if tgt_to_src[j] == i {
            pairs.push((i, j));
            if pairs.len() >= max_corr {
                break;
            }
        }
    }
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <source_xyz> <target_xyz>", args[0]);
        std::process::exit(1);
    }

    let src_pts = load_xyz(&args[1])?;
    let tgt_pts = load_xyz(&args[2])?;

    if src_pts.len() < 3 || tgt_pts.len() < 3 {
        eprintln!("Point clouds must each have at least 3 points");
        std::process::exit(1);
    }

    // Build simple mutual-NN correspondences (analogous to KISS matcher’s initial pass).
    let pairs = mutual_nn_pairs(&src_pts, &tgt_pts, 5000);
    if pairs.len() < 3 {
        eprintln!("Not enough correspondences after mutual NN");
        std::process::exit(1);
    }

    let n = pairs.len();
    let mut data = inlier::types::DataMatrix::zeros(n, 6);
    for (row, (i, j)) in pairs.into_iter().enumerate() {
        let s = src_pts[i];
        let t = tgt_pts[j];
        data.set(row, 0, s[0]);
        data.set(row, 1, s[1]);
        data.set(row, 2, s[2]);
        data.set(row, 3, t[0]);
        data.set(row, 4, t[1]);
        data.set(row, 5, t[2]);
    }

    let settings = MetasacSettings {
        min_iterations: 2000,
        max_iterations: 5000,
        confidence: 0.9999,
        ..MetasacSettings::default()
    };

    let pipeline = teaser_pointcloud_registration_pipeline(
        0.05,  // inlier threshold
        0.05,  // noise bound beta
        1.0,   // c_bar (unused in this simplified pipeline)
        false, // has_priors
        None,  // rng_seed
        settings, 0.2, // suppression radius
        5,   // min neighbors
        0.8, // linearity threshold
        5,   // k-core min degree
    );

    if let Some(result) = pipeline.run(&data) {
        println!("Estimated scale: {:.4}", result.model.scale);
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
