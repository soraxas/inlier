//! Automatic parameter estimation from point cloud noise and density statistics.
//!
//! [`auto_tune_settings`] samples ~2 000 representative points, computes 20-NN
//! for each, and derives two scalar statistics:
//!
//! - **`noise_sigma`** — median of per-point plane residuals.  Each residual is
//!   `|n · p + d|` where `(n, d)` is the PCA plane fitted to the point's 20
//!   neighbours.  The median is robust to outliers and approximates the sensor
//!   measurement noise floor.
//!
//! - **`point_spacing`** — median nearest-neighbour distance.  Approximates the
//!   surface point density without requiring a full range image.
//!
//! From these two statistics it derives all downstream thresholds:
//!
//! | Threshold | Formula | Rationale |
//! |-----------|---------|-----------|
//! | `dist_thresh` | `noise_sigma × 4` | RANSAC inlier band at 4σ |
//! | `angle_thresh` | `15° + σ × 200°` | Looser for noisier sensors |
//! | `min_cluster_size` | `0.5 m² × density` | ~half a square metre |
//! | `merge_angle_thresh` | `angle_thresh × 0.4` | Joins same-wall fragments |
//! | `merge_dist_thresh` | `dist_thresh × 3` | Accounts for wall offsets |
//! | `merge_min_pts` | `min_cluster_size / 4` | Loose minimum after merge |
//! | `grow_dist_thresh` | `dist_thresh × 2` | Mops up near-plane leftovers |

use crate::spatial_grid::{estimate_cell_size, build_grid, knn};
use crate::normals::pca_normal_and_curvature;

/// Algorithm parameter set derived by [`auto_tune_settings`].
///
/// All angle fields are in **degrees** (matching the UI slider convention).
/// All distance fields are in the same units as the input point cloud.
#[derive(Debug, Clone)]
pub struct TunedSettings {
    /// RANSAC inlier distance band (= `noise_sigma × 4`).
    pub dist_thresh: f32,
    /// Max normal deviation for region growing, in degrees (= `15 + σ×200`, clamped).
    pub angle_thresh: f32,
    /// Minimum region-growing cluster size to run RANSAC on.
    pub min_cluster_size: usize,
    /// Max normal angle for merge co-planarity test, in degrees (= `angle_thresh × 0.4`).
    pub merge_angle_thresh: f32,
    /// Max plane-offset difference for merge (= `dist_thresh × 3`).
    pub merge_dist_thresh: f32,
    /// Minimum inlier count after merge (= `min_cluster_size / 4`).
    pub merge_min_pts: usize,
    /// Max point-to-plane distance for grow step (= `dist_thresh × 2`).
    pub grow_dist_thresh: f32,
    /// Human-readable summary for display in the status bar.
    pub description: String,
}

/// Estimate good algorithm parameters from point cloud statistics.
///
/// Strides the input to at most 2 000 sample points for speed, builds a local
/// spatial grid, then computes 20-NN for each sample.
///
/// # Example
/// ```no_run
/// # use spatialrust_inlier::auto_tune::auto_tune_settings;
/// # let pts: Vec<[f32; 3]> = vec![];
/// let settings = auto_tune_settings(&pts);
/// println!("dist_thresh = {:.3}", settings.dist_thresh);
/// println!("{}", settings.description);
/// ```
pub fn auto_tune_settings(pts: &[[f32; 3]]) -> TunedSettings {
    let n = pts.len();
    // Stride to at most 2000 samples for speed.
    let stride = (n / 2000).max(1);
    let samples: Vec<[f32; 3]> = pts.iter().step_by(stride).cloned().collect();
    let ns = samples.len();

    let cell_size = estimate_cell_size(&samples);
    let grid = build_grid(&samples, cell_size);

    let k = 20usize;
    let mut residuals: Vec<f32> = Vec::with_capacity(ns);
    let mut nn_dists: Vec<f32> = Vec::with_capacity(ns);

    for i in 0..ns {
        let neighbors = knn(&samples, i, k, cell_size, &grid);
        if neighbors.len() < 6 {
            continue;
        }

        // Nearest-neighbour distance (proxy for point spacing).
        let p = samples[i];
        let mut min_d = f32::MAX;
        for &j in &neighbors {
            let q = samples[j];
            let d = ((p[0] - q[0]).powi(2)
                + (p[1] - q[1]).powi(2)
                + (p[2] - q[2]).powi(2))
                .sqrt();
            if d < min_d {
                min_d = d;
            }
        }
        if min_d < f32::MAX {
            nn_dists.push(min_d);
        }

        // Local plane residual.
        if let Some((normal, _)) = pca_normal_and_curvature(&samples, &neighbors) {
            let cx = neighbors.iter().map(|&j| samples[j][0]).sum::<f32>()
                / neighbors.len() as f32;
            let cy = neighbors.iter().map(|&j| samples[j][1]).sum::<f32>()
                / neighbors.len() as f32;
            let cz = neighbors.iter().map(|&j| samples[j][2]).sum::<f32>()
                / neighbors.len() as f32;
            let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
            let res = (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + d).abs();
            residuals.push(res);
        }
    }

    // Median (sort in-place, take middle element).
    let median = |v: &mut Vec<f32>| -> f32 {
        if v.is_empty() {
            return 0.01;
        }
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    let noise_sigma = median(&mut residuals).max(1e-4);
    let point_spacing = median(&mut nn_dists).max(1e-4);

    let dist_thresh = (noise_sigma * 4.0).clamp(0.01, 0.5);
    // Looser normal angle for noisier sensors.
    let angle_thresh = (15.0_f32 + noise_sigma * 200.0).clamp(5.0, 45.0);
    // ~0.5 m² of surface at the estimated density.
    let density = 1.0 / (point_spacing * point_spacing);
    let min_cluster_size = ((density * 0.5) as usize).clamp(20, 2000);

    let merge_angle_thresh = (angle_thresh * 0.4).clamp(2.0, 10.0);
    let merge_dist_thresh = (dist_thresh * 3.0).clamp(0.05, 1.0);
    let merge_min_pts = (min_cluster_size / 4).clamp(50, 1000);
    let grow_dist_thresh = (dist_thresh * 2.0).clamp(0.02, 1.0);

    TunedSettings {
        dist_thresh,
        angle_thresh,
        min_cluster_size,
        merge_angle_thresh,
        merge_dist_thresh,
        merge_min_pts,
        grow_dist_thresh,
        description: format!(
            "Auto-tune: noise_σ={noise_sigma:.4}m  spacing={point_spacing:.4}m \
             → dist={dist_thresh:.3}  angle={angle_thresh:.1}°  \
             min_pts={min_cluster_size}  grow_dist={grow_dist_thresh:.3}",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_auto_tune() {
        // Flat grid of points: near-zero noise_sigma, small grow_dist_thresh.
        let pts: Vec<[f32; 3]> = (0..400)
            .map(|i| [(i % 20) as f32 * 0.05, (i / 20) as f32 * 0.05, 0.0])
            .collect();
        let t = auto_tune_settings(&pts);
        assert!(t.dist_thresh > 0.0, "dist_thresh must be positive");
        assert!(t.grow_dist_thresh >= t.dist_thresh, "grow must be >= seg dist");
        assert!(t.merge_dist_thresh >= t.dist_thresh, "merge must be >= seg dist");
        assert!(!t.description.is_empty());
    }
}
