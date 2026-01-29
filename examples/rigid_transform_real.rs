//! Example: Rigid transform estimation using real-world correspondences.

use inlier::estimate_rigid_transform_with_callback;
use inlier::models::RigidTransform;
use inlier::scoring::Score;
use inlier::{RansacCallback, RansacCallbackStage};
use nalgebra::{DMatrix, Matrix3, Vector3, Matrix4};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rigid Transform (Real Data) ===\n");

    let data_path = "data/rigid_pose_example_points.txt";
    let data = load_table(data_path, 6)?;
    let n = data.len();
    let gt = load_pose4x4("data/rigid_pose_example_gt.txt")?;

    let mut points1 = DMatrix::<f64>::zeros(n, 3);
    let mut points2 = DMatrix::<f64>::zeros(n, 3);
    for (i, row) in data.iter().enumerate() {
        points1[(i, 0)] = row[0];
        points1[(i, 1)] = row[1];
        points1[(i, 2)] = row[2];
        points2[(i, 0)] = row[3];
        points2[(i, 1)] = row[4];
        points2[(i, 2)] = row[5];
    }

    println!("Loaded {} correspondences from {}", n, data_path);

    let mut progress = ProgressPrinter { print_every: 25 };
    let threshold = 0.05;
    let result = estimate_rigid_transform_with_callback(
        &points1,
        &points2,
        threshold,
        None,
        &mut progress,
    )?;

    println!("Estimation results:");
    println!("  Inliers: {}", result.inliers.len());
    println!("  Score: {:?}", result.score);
    println!("  Iterations: {}", result.iterations);

    println!("\nEstimated transform:");
    println!("  R: {:?}", result.model.rotation);
    println!("  t: {:?}", result.model.translation);

    let r_gt = gt.fixed_view::<3, 3>(0, 0).into_owned();
    let t_gt = Vector3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
    let r_est = result.model.rotation.to_rotation_matrix();
    let t_est = result.model.translation.vector;
    let angle = rotation_error_rad(r_gt, *r_est.matrix());
    let t_err = (t_est - t_gt).norm();

    println!("\nTransform error vs ground truth:");
    println!("  rotation error: {:.4} rad", angle);
    println!("  translation error: {:.4}", t_err);

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

fn load_pose4x4(path: &str) -> Result<Matrix4<f64>, Box<dyn std::error::Error>> {
    let rows = load_table(path, 4)?;
    if rows.len() != 4 {
        return Err(format!("Expected 4 rows in {}", path).into());
    }
    Ok(Matrix4::from_row_slice(&[
        rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[1][0], rows[1][1], rows[1][2],
        rows[1][3], rows[2][0], rows[2][1], rows[2][2], rows[2][3], rows[3][0], rows[3][1],
        rows[3][2], rows[3][3],
    ]))
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

impl RansacCallback<RigidTransform, Score> for ProgressPrinter {
    fn on_stage(
        &mut self,
        stage: RansacCallbackStage,
        iteration: usize,
        _best_model: &Option<RigidTransform>,
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
