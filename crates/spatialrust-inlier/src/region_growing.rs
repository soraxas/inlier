//! Ling et al. ISPRS 2024 region-growing + RANSAC plane segmentation.
//!
//! ## Pipeline (three steps)
//!
//! ```text
//! Step 1 — Normal + curvature estimation
//!   For each point: query 27-cell grid neighbourhood, fit covariance,
//!   extract smallest eigenvector (normal) and curvature = λ_min / trace.
//!
//! Step 2 — Region growing
//!   Sort points by curvature ascending (flattest = best seed).
//!   BFS-grow each unvisited seed: accept neighbour when
//!     |n_cur · n_nb| > cos(angle_thresh).
//!   Discard clusters below min_cluster_size.
//!
//! Step 3 — RANSAC per cluster + aggregation sweep
//!   Fit a plane to each cluster via Simple / MSAC / MAGSAC++.
//!   Accept cluster points within dist_thresh as initial inliers.
//!   Aggregation: sweep ALL unassigned points; absorb those within
//!     dist_thresh whose local normal also agrees within angle_thresh.
//! ```
//!
//! The output is a list of `(unit_normal, d, inlier_indices)` tuples sorted by
//! inlier count descending.  The plane equation is `normal · p + d ≈ 0`.
//!
//! ## RANSAC modes
//!
//! | Variant | Algorithm | Use when |
//! |---------|-----------|----------|
//! | [`RansacMode::Simple`] | 3-pt RANSAC + LS refine | Fast, no deps |
//! | [`RansacMode::Msac`] | MSAC + IRLS | More accurate, needs `inlier` crate |
//! | [`RansacMode::Magsac`] | MAGSAC++ σ-consensus | Mixed-noise scans; slowest |

use std::collections::VecDeque;

use crate::spatial_grid::{estimate_cell_size, build_grid, knn};
use crate::normals::{pca_normal_and_curvature, fit_plane_3pts, fit_plane_ls};

/// Which RANSAC scorer to use when fitting a plane to each region-growing cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RansacMode {
    /// Pure-Rust 3-point RANSAC with least-squares refinement on inliers.
    /// Fast, zero dependencies beyond the `normals` module.
    Simple,
    /// inlier-crate MSAC with IRLS local optimisation.
    /// Softer inlier weighting gives more accurate planes on noisy data.
    Msac,
    /// MAGSAC++ σ-consensus (threshold-free, marginalises noise scale).
    /// Best for mixed-noise scans (e.g. outdoor LiDAR mixed with structured-light).
    Magsac,
}

/// Run the full Ling et al. 2024 three-step pipeline on a raw point cloud.
///
/// # Parameters
/// - `pts` — flat `[x, y, z]` slice (any coordinate system).
/// - `k` — neighbourhood size for normal/curvature estimation (default 20).
/// - `angle_thresh` — **radians**; max normal deviation to grow into a neighbour
///   (default ~10°).
/// - `min_cluster_size` — discard region-growing clusters smaller than this.
/// - `dist_thresh` — RANSAC inlier distance band (same units as `pts`).
/// - `mode` — which RANSAC scorer to use.
/// - `sigma_max` — MAGSAC++ σ_max (multiples of `dist_thresh`; ignored for Simple/MSAC).
/// - `max_iterations` / `confidence` — MSAC + MAGSAC++ iteration budget.
///
/// # Returns
/// `Vec<(unit_normal, d, inlier_indices)>` sorted by inlier count descending.
/// Plane equation: `normal · p + d ≈ 0`.
///
/// Returns an empty vec if `pts.len() < 3`.
#[allow(clippy::too_many_arguments)]
pub fn region_growing_ransac(
    pts: &[[f32; 3]],
    k: usize,
    angle_thresh: f32,
    min_cluster_size: usize,
    dist_thresh: f32,
    mode: RansacMode,
    sigma_max: f64,
    max_iterations: usize,
    confidence: f64,
) -> Vec<([f32; 3], f32, Vec<usize>)> {
    let n = pts.len();
    if n < 3 {
        return vec![];
    }

    let cell_size = estimate_cell_size(pts);
    let grid = build_grid(pts, cell_size);

    // Step 1: per-point normals and curvatures.
    let mut normals: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    let mut curvatures: Vec<f32> = vec![f32::MAX; n];
    for i in 0..n {
        let neighbors = knn(pts, i, k, cell_size, &grid);
        if neighbors.len() < 3 {
            continue;
        }
        if let Some((nv, cv)) = pca_normal_and_curvature(pts, &neighbors) {
            normals[i] = nv;
            curvatures[i] = cv;
        }
    }

    // Step 2: region growing sorted by curvature ascending (flattest = best seed).
    let cos_thresh = angle_thresh.cos();
    let mut visited = vec![false; n];
    let mut sorted_idx: Vec<usize> = (0..n)
        .filter(|&i| curvatures[i] < f32::MAX)
        .collect();
    sorted_idx.sort_unstable_by(|&a, &b| {
        curvatures[a].partial_cmp(&curvatures[b]).unwrap()
    });

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for &seed in &sorted_idx {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;

        let mut cluster = vec![seed];
        let mut queue = VecDeque::new();
        queue.push_back(seed);

        while let Some(cur) = queue.pop_front() {
            for nb in knn(pts, cur, k, cell_size, &grid) {
                if visited[nb] {
                    continue;
                }
                let nc = normals[cur];
                let nn = normals[nb];
                let dot = (nc[0] * nn[0] + nc[1] * nn[1] + nc[2] * nn[2]).abs();
                if dot < cos_thresh {
                    continue;
                }
                visited[nb] = true;
                cluster.push(nb);
                queue.push_back(nb);
            }
        }

        if cluster.len() >= min_cluster_size {
            clusters.push(cluster);
        }
    }

    // Step 3: RANSAC per cluster + aggregation sweep.
    let mut assigned = vec![false; n];
    let mut result: Vec<([f32; 3], f32, Vec<usize>)> = Vec::new();

    for cluster in clusters {
        let unassigned: Vec<usize> = cluster
            .iter()
            .cloned()
            .filter(|&i| !assigned[i])
            .collect();
        if unassigned.len() < min_cluster_size {
            continue;
        }

        let cluster_pts: Vec<[f32; 3]> = unassigned.iter().map(|&i| pts[i]).collect();

        let msac_settings = || {
            Some(inlier::MetasacSettings {
                max_iterations,
                confidence,
                ..inlier::MetasacSettings::default()
            })
        };

        let fit = match mode {
            RansacMode::Simple => ransac_plane_simple(&cluster_pts, dist_thresh, 200, 42),
            RansacMode::Msac => {
                crate::plane::fit_plane_msac(
                    &cluster_pts,
                    dist_thresh as f64,
                    msac_settings(),
                )
                .map(|(n, d, _)| (n, d))
            }
            RansacMode::Magsac => {
                crate::plane::fit_plane_magsac_raw(
                    &cluster_pts,
                    sigma_max,
                    msac_settings(),
                )
                .map(|(n, d, _)| (n, d))
            }
        };

        let (normal, d) = match fit {
            Some(nd) => nd,
            None => continue,
        };

        let mut plane_pts: Vec<usize> = Vec::new();
        for &gi in &unassigned {
            let dist =
                (normal[0] * pts[gi][0] + normal[1] * pts[gi][1] + normal[2] * pts[gi][2] + d).abs();
            if dist < dist_thresh {
                plane_pts.push(gi);
                assigned[gi] = true;
            }
        }
        if plane_pts.len() < min_cluster_size {
            continue;
        }

        // Aggregation sweep: absorb unassigned points that pass distance + normal.
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let dist =
                (normal[0] * pts[i][0] + normal[1] * pts[i][1] + normal[2] * pts[i][2] + d).abs();
            if dist >= dist_thresh {
                continue;
            }
            let dot = (normals[i][0] * normal[0]
                + normals[i][1] * normal[1]
                + normals[i][2] * normal[2])
                .abs();
            if dot < cos_thresh {
                continue;
            }
            plane_pts.push(i);
            assigned[i] = true;
        }

        result.push((normal, d, plane_pts));
    }

    result.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
    result
}

/// 3-point RANSAC plane fitter with least-squares refinement on inliers.
///
/// Uses a 64-bit LCG (`s' = s × 6364136223846793005 + 1442695040888963407`) to
/// sample triples without external RNG dependencies.  After finding the best
/// hypothesis, refits via PCA on all inliers.
///
/// Returns `(unit_normal, d)` or `None` if fewer than 3 inliers found.
pub fn ransac_plane_simple(
    pts: &[[f32; 3]],
    threshold: f32,
    max_iters: usize,
    seed: u64,
) -> Option<([f32; 3], f32)> {
    let n = pts.len();
    if n < 3 {
        return None;
    }

    let mut rng = seed;
    let mut lcg = move || -> usize {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };

    let mut best_count = 0usize;
    let mut best_normal = [0f32; 3];
    let mut best_d = 0f32;

    for _ in 0..max_iters {
        let i0 = lcg() % n;
        let i1 = lcg() % n;
        let i2 = lcg() % n;
        if i0 == i1 || i0 == i2 || i1 == i2 {
            continue;
        }

        let (normal, d) = match fit_plane_3pts(pts[i0], pts[i1], pts[i2]) {
            Some(nd) => nd,
            None => continue,
        };

        let count = pts
            .iter()
            .filter(|&&p| {
                (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + d).abs() < threshold
            })
            .count();

        if count > best_count {
            best_count = count;
            best_normal = normal;
            best_d = d;
        }
    }

    if best_count < 3 {
        return None;
    }

    // Least-squares refinement on inliers.
    let inliers: Vec<[f32; 3]> = pts
        .iter()
        .cloned()
        .filter(|&p| {
            (best_normal[0] * p[0]
                + best_normal[1] * p[1]
                + best_normal[2] * p[2]
                + best_d)
                .abs()
                < threshold
        })
        .collect();

    fit_plane_ls(&inliers).or(Some((best_normal, best_d)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normals::normalize3;

    fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Generate a synthetic multi-plane cloud (same RNG as the viewer demo).
    fn synthetic_multi_plane(
        n_planes: usize,
        n_inliers: usize,
        noise_std: f32,
        n_outliers: usize,
        seed: u32,
    ) -> Vec<[f32; 3]> {
        let mut s = seed as u64;
        let mut rng = move || -> f32 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let mut out = Vec::new();
        for pi in 0..n_planes.min(8) {
            let normal = loop {
                let (x, y, z) = (rng(), rng(), rng());
                let len = (x * x + y * y + z * z).sqrt();
                if len > 0.01 && len <= 1.0 {
                    break normalize3([x, y, z]);
                }
            };
            let offset = pi as f32 * 2.5;
            let p0 = [
                -offset * normal[0],
                -offset * normal[1],
                -offset * normal[2],
            ];
            let up = if normal[2].abs() < 0.9 {
                [0f32, 0., 1.]
            } else {
                [1., 0., 0.]
            };
            let t1 = normalize3(cross3(normal, up));
            let t2 = cross3(normal, t1);
            for _ in 0..n_inliers {
                let u = rng() * 3.0;
                let v = rng() * 3.0;
                let noise = rng() * noise_std;
                out.push([
                    p0[0] + u * t1[0] + v * t2[0] + noise * normal[0],
                    p0[1] + u * t1[1] + v * t2[1] + noise * normal[1],
                    p0[2] + u * t1[2] + v * t2[2] + noise * normal[2],
                ]);
            }
        }
        for _ in 0..n_outliers {
            out.push([rng() * 4., rng() * 4., rng() * 4.]);
        }
        out
    }

    #[test]
    fn smoke_3_planes_simple() {
        let pts = synthetic_multi_plane(3, 600, 0.03, 200, 42);
        let planes = region_growing_ransac(
            &pts,
            20,
            10f32.to_radians(),
            30,
            0.08,
            RansacMode::Simple,
            0.0,
            1000,
            0.99,
        );
        assert!(planes.len() >= 2, "expected ≥2 planes, got {}", planes.len());
        for (i, (_, _, inliers)) in planes.iter().enumerate() {
            assert!(
                inliers.len() >= 200,
                "plane {} has {} inliers (want ≥200)",
                i + 1,
                inliers.len()
            );
        }
    }
}
