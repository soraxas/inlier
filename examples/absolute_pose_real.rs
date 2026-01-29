//! Example: Absolute pose estimation using real-world correspondences.

use inlier::estimate_absolute_pose_with_callback;
use inlier::models::AbsolutePose;
use inlier::scoring::Score;
use inlier::settings::{LocalOptimizationType, RansacSettings, SamplerType, ScoringType};
use inlier::{RansacCallback, RansacCallbackStage};
use nalgebra::{DMatrix, Matrix3, Vector3};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Absolute Pose (Real Data) ===\n");

    let data_path = "data/pose6dscene_points.txt";
    let data = load_table(data_path, 5)?;
    let n = data.len();

    let intrinsics = load_matrix3("data/pose6dscene.K")?;
    let gt_pose = load_pose3x4("data/pose6dscene_gt.txt")?;

    let fx = intrinsics[(0, 0)];
    let fy = intrinsics[(1, 1)];
    let cx = intrinsics[(0, 2)];
    let cy = intrinsics[(1, 2)];

    let mut points_2d = DMatrix::<f64>::zeros(n, 2);
    let mut points_3d = DMatrix::<f64>::zeros(n, 3);
    for (i, row) in data.iter().enumerate() {
        let u = row[0];
        let v = row[1];
        points_2d[(i, 0)] = (u - cx) / fx;
        points_2d[(i, 1)] = (v - cy) / fy;
        points_3d[(i, 0)] = row[2];
        points_3d[(i, 1)] = row[3];
        points_3d[(i, 2)] = row[4];
    }

    println!("Loaded {} correspondences from {}", n, data_path);

    let mut progress = ProgressPrinter { print_every: 25 };
    let threshold_px = 8.0;
    let threshold = threshold_px / fx.max(fy);
    let mut settings = RansacSettings::default();
    settings.sampler = SamplerType::Uniform;
    settings.scoring = ScoringType::Ransac;
    settings.local_optimization = LocalOptimizationType::None;
    settings.final_optimization = LocalOptimizationType::None;
    settings.min_iterations = 1000;
    settings.max_iterations = 20000;
    let result = estimate_absolute_pose_with_callback(
        &points_3d,
        &points_2d,
        threshold,
        Some(settings),
        &mut progress,
    )?;

    println!("Estimation results:");
    println!("  Inliers: {}", result.inliers.len());
    println!("  Score: {:?}", result.score);
    println!("  Iterations: {}", result.iterations);

    println!("\nEstimated pose (rotation + translation):");
    println!("  R: {:?}", result.model.rotation);
    println!("  t: {:?}", result.model.translation);

    let (r_gt, t_gt) = gt_pose;
    let r_est = result.model.rotation.to_rotation_matrix();
    let t_est = result.model.translation.vector;
    let angle = rotation_error_rad(r_gt, *r_est.matrix());
    let t_err = (t_est - t_gt).norm();

    println!("\nPose error vs ground truth:");
    println!("  rotation error: {:.4} rad", angle);
    println!("  translation error: {:.4}", t_err);
    println!("\nNote:");
    println!("  For stronger results, enable the `p3p` or `kornia-pnp` features.");

    Ok(())
}

fn load_table(path: &str, cols: usize) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != cols {
            return Err(format!("Expected {} columns in {}", cols, path).into());
        }
        rows.push(values);
    }
    Ok(rows)
}

fn load_matrix3(path: &str) -> Result<Matrix3<f64>, Box<dyn std::error::Error>> {
    let rows = load_table(path, 3)?;
    if rows.len() != 3 {
        return Err(format!("Expected 3 rows in {}", path).into());
    }
    Ok(Matrix3::from_row_slice(&[
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    ]))
}

fn load_pose3x4(path: &str) -> Result<(Matrix3<f64>, Vector3<f64>), Box<dyn std::error::Error>> {
    let rows = load_table(path, 4)?;
    if rows.len() != 3 {
        return Err(format!("Expected 3 rows in {}", path).into());
    }
    let r = Matrix3::from_row_slice(&[
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    ]);
    let t = Vector3::new(rows[0][3], rows[1][3], rows[2][3]);
    Ok((r, t))
}

fn rotation_error_rad(r_gt: Matrix3<f64>, r_est: Matrix3<f64>) -> f64 {
    let r = r_gt.transpose() * r_est;
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos()
}

struct ProgressPrinter {
    print_every: usize,
}

impl RansacCallback<AbsolutePose, Score> for ProgressPrinter {
    fn on_stage(
        &mut self,
        stage: RansacCallbackStage,
        iteration: usize,
        _best_model: &Option<AbsolutePose>,
        best_score: &Option<Score>,
        _best_inliers: &[usize],
    ) {
        if iteration % self.print_every == 0 {
            if let RansacCallbackStage::Iteration = stage {
                if let Some(score) = best_score {
                    println!("  iter {:4}: best score {:?}", iteration, score);
                }
            }
        }
    }
}
