//! Post-processing operations on a set of segmented planes:
//! union-find merge and iterative grow with furniture filters.
//!
//! ## Merge
//!
//! Region growing + RANSAC tends to over-segment large surfaces into several
//! smaller planes separated by occlusions, doorways, or noise.  [`merge_planes`]
//! uses a union-find algorithm to collapse fragments that describe the same
//! physical surface (nearly parallel normals, similar plane offset), then refits
//! each merged group via PCA.
//!
//! ## Grow
//!
//! After merge, many wall/floor/ceiling fragments remain unassigned because
//! region growing was seeded from flat areas and stopped at noise boundaries.
//! [`grow_planes`] sweeps leftover points and assigns each to the nearest
//! geometrically matching plane, then refits via PCA.  Repeating this up to
//! `grow_max_iters` times lets reachable fragments accumulate across passes
//! while the plane is continuously refit to stay accurate.
//!
//! ### Furniture filters (Rabbani et al. 2006 / PCL region growing)
//!
//! A plain distance-only absorb pulls in furniture close to walls.  Three
//! independently toggleable filters guard against this:
//!
//! | Filter | Rejects | Limitation |
//! |--------|---------|------------|
//! | Normal | Point's local normal deviates from plane normal by more than `normal_angle` | Flush furniture (bookcase flush against wall) bypasses it |
//! | Curvature | Point's local curvature exceeds `max_curvature` (edges, rounded surfaces) | Can over-reject wall corners |
//! | Connectivity | No 27-cell grid neighbour of the point is an existing plane inlier | Furniture touching a wall passes this filter |
//!
//! The inlier sets for the connectivity filter are built **before** the grow pass
//! so a furniture chain cannot relay across the cloud point-by-point during a
//! single iteration.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::normals::pca_normal_and_curvature;
use crate::spatial_grid::{build_grid, estimate_cell_size, knn};

/// Parameters for the grow step.
///
/// All fields are public so callers can construct the struct with struct-literal
/// syntax (no builder pattern needed for a flat config).
#[derive(Debug, Clone)]
pub struct GrowArgs {
    /// Maximum point-to-plane distance to absorb a leftover point (same units as pts).
    pub dist_thresh: f32,
    /// Enable normal agreement filter.
    pub use_normal: bool,
    /// Cosine of the maximum allowed angle between the candidate point's local
    /// surface normal and the target plane normal (`cos(grow_normal_angle_radians)`).
    pub normal_cos_thresh: f32,
    /// Enable curvature filter.
    pub use_curvature: bool,
    /// Maximum local curvature (`λ_min / trace`) to allow absorption.
    /// Flat surfaces ≈ 0; furniture edges / curves > 0.05.
    pub max_curvature: f32,
    /// Enable spatial connectivity filter (point must neighbour an existing inlier).
    pub use_connectivity: bool,
}

/// Collapse co-planar plane fragments via union-find and refit via PCA.
///
/// Two planes `i` and `j` are merged when **both** conditions hold:
/// - `|n_i · n_j| > cos(angle_thresh)` — nearly parallel normals.
/// - `|d_i − d_j| < dist_thresh` — same signed offset from origin
///   (d_j is sign-adjusted to match the n_i orientation before comparing).
///
/// Union-find with path compression collects transitively connected groups.
/// Each merged group is refit via PCA on the union of all inlier points, then
/// dropped if the combined count is below `min_pts`.
///
/// # Parameters
/// - `planes` — `(unit_normal, d, inlier_indices)` list from
///   [`region_growing_ransac`](crate::region_growing::region_growing_ransac).
/// - `all_pts` — full point cloud used to dereference inlier indices.
/// - `angle_thresh` — **radians**; max normal deviation for co-planarity (default ~5°).
/// - `dist_thresh` — max plane-offset difference (default 0.15 m).
/// - `min_pts` — drop merged plane if fewer inliers.
///
/// Returns planes sorted by inlier count descending.
pub fn merge_planes(
    planes: &[([f32; 3], f32, Vec<usize>)],
    all_pts: &[[f32; 3]],
    angle_thresh: f32,
    dist_thresh: f32,
    min_pts: usize,
) -> Vec<([f32; 3], f32, Vec<usize>)> {
    let n = planes.len();
    if n == 0 {
        return vec![];
    }

    let cos_thresh = angle_thresh.cos();

    // Union-find with path halving.  Using a standalone fn avoids the double
    // mutable borrow that a closure over `parent` would create when the caller
    // also writes to `parent[ri]` on the same line the closure is still live.
    fn uf_find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    let mut parent: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let ni = planes[i].0;
            let nj = planes[j].0;
            let dot = (ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2]).abs();
            if dot < cos_thresh {
                continue;
            }
            // Adjust d_j sign so both offsets reference the same normal orientation.
            let di = planes[i].1;
            let same_sign = planes[i].0[0] * planes[j].0[0]
                + planes[i].0[1] * planes[j].0[1]
                + planes[i].0[2] * planes[j].0[2]
                >= 0.0;
            let dj = if same_sign { planes[j].1 } else { -planes[j].1 };
            if (di - dj).abs() > dist_thresh {
                continue;
            }
            let ri = uf_find(&mut parent, i);
            let rj = uf_find(&mut parent, j);
            if ri != rj {
                parent[ri] = rj;
            }
        }
    }

    // Collect groups.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    let mut result = Vec::new();
    for (_, members) in groups {
        let mut all_inliers: Vec<usize> = members
            .iter()
            .flat_map(|&pi| planes[pi].2.iter().cloned())
            .collect();
        all_inliers.sort_unstable();
        all_inliers.dedup();
        if all_inliers.len() < min_pts {
            continue;
        }

        // Refit via PCA on combined inlier points.
        let pts3d: Vec<[f32; 3]> = all_inliers.iter().map(|&i| all_pts[i]).collect();
        let idx: Vec<usize> = (0..pts3d.len()).collect();
        let (normal, _) = match pca_normal_and_curvature(&pts3d, &idx) {
            Some(r) => r,
            None => continue,
        };
        let n_f = pts3d.len() as f32;
        let cx = pts3d.iter().map(|p| p[0]).sum::<f32>() / n_f;
        let cy = pts3d.iter().map(|p| p[1]).sum::<f32>() / n_f;
        let cz = pts3d.iter().map(|p| p[2]).sum::<f32>() / n_f;
        let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
        result.push((normal, d, all_inliers));
    }

    result.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
    result
}

/// Absorb unassigned leftover points into their nearest geometrically matching plane.
///
/// One call performs a single grow-refit pass.  To iterate until convergence,
/// call `grow_planes` in a loop and stop when the total inlier count stops
/// increasing (see [`DollhouseParams::grow_max_iters`](crate::dollhouse::DollhouseParams)).
///
/// ## Algorithm
///
/// 1. Mark all existing inlier points as assigned.
/// 2. If any furniture filter is active, build a spatial grid and compute
///    per-point normals/curvatures for all *unassigned* points.
/// 3. For each unassigned point, find the nearest plane within `args.dist_thresh`
///    that also passes all active furniture filters.
/// 4. Refit each plane via PCA on its expanded inlier set.
///
/// ## Connectivity filter detail
///
/// The per-plane inlier `HashSet`s are built from the **initial inliers only**
/// (before any absorption in this call).  This prevents a furniture chain from
/// relaying across the cloud one point at a time during a single pass.
///
/// # Returns
/// Updated `(normal, d, inlier_indices)` tuples sorted by inlier count descending.
/// Planes that end up with fewer than 3 inliers are dropped.
pub fn grow_planes(
    planes: &[([f32; 3], f32, Vec<usize>)],
    all_pts: &[[f32; 3]],
    args: &GrowArgs,
) -> Vec<([f32; 3], f32, Vec<usize>)> {
    let n_pts = all_pts.len();
    let mut assigned = vec![false; n_pts];
    let mut extended: Vec<Vec<usize>> = planes
        .iter()
        .map(|(_, _, v)| {
            for &i in v {
                if i < n_pts {
                    assigned[i] = true;
                }
            }
            v.clone()
        })
        .collect();

    // Precompute per-point normals + curvatures (needed by normal/curvature filters).
    let need_normals = args.use_normal || args.use_curvature;
    let (pt_normals, pt_curvatures, grid, cell_size) = if need_normals {
        let cs = estimate_cell_size(all_pts);
        let g = build_grid(all_pts, cs);
        let mut normals = vec![[0f32; 3]; n_pts];
        let mut curvatures = vec![f32::MAX; n_pts];
        for i in 0..n_pts {
            if assigned[i] {
                continue; // skip already-assigned points
            }
            let neighbors = knn(all_pts, i, 20, cs, &g);
            if let Some((nv, cv)) = pca_normal_and_curvature(all_pts, &neighbors) {
                normals[i] = nv;
                curvatures[i] = cv;
            }
        }
        (normals, curvatures, g, cs)
    } else {
        let cs = estimate_cell_size(all_pts);
        // Build grid even without normal/curvature if connectivity filter is on.
        let g = if args.use_connectivity {
            build_grid(all_pts, cs)
        } else {
            HashMap::new()
        };
        (vec![[0f32; 3]; n_pts], vec![f32::MAX; n_pts], g, cs)
    };

    // Per-plane inlier HashSets for connectivity lookup (initial inliers only).
    let inlier_sets: Vec<HashSet<usize>> = if args.use_connectivity {
        planes
            .iter()
            .map(|(_, _, v)| v.iter().cloned().collect())
            .collect()
    } else {
        vec![]
    };

    let inv = 1.0 / cell_size;

    for (idx, &pt) in all_pts.iter().enumerate() {
        if assigned[idx] {
            continue;
        }

        // Curvature gate (cheap, check before the per-plane loop).
        if args.use_curvature && pt_curvatures[idx] > args.max_curvature {
            continue;
        }

        let mut best_dist = args.dist_thresh;
        let mut best_plane: Option<usize> = None;

        for (pi, (pn, d, _)) in planes.iter().enumerate() {
            // Distance gate.
            let dist = (pn[0] * pt[0] + pn[1] * pt[1] + pn[2] * pt[2] + d).abs();
            if dist >= best_dist {
                continue;
            }

            // Normal agreement gate.
            if args.use_normal {
                let dot = (pt_normals[idx][0] * pn[0]
                    + pt_normals[idx][1] * pn[1]
                    + pt_normals[idx][2] * pn[2])
                    .abs();
                if dot < args.normal_cos_thresh {
                    continue;
                }
            }

            // Connectivity gate: at least one 27-cell neighbour must be an existing inlier.
            if args.use_connectivity {
                let cx = (pt[0] * inv).floor() as i32;
                let cy = (pt[1] * inv).floor() as i32;
                let cz = (pt[2] * inv).floor() as i32;
                let mut connected = false;
                'conn: for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dz in -1i32..=1 {
                            if let Some(cell) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &nb in cell {
                                    if inlier_sets[pi].contains(&nb) {
                                        connected = true;
                                        break 'conn;
                                    }
                                }
                            }
                        }
                    }
                }
                if !connected {
                    continue;
                }
            }

            best_dist = dist;
            best_plane = Some(pi);
        }

        if let Some(pi) = best_plane {
            extended[pi].push(idx);
        }
    }

    // Refit each plane via PCA on its expanded inlier set.
    let mut result = Vec::with_capacity(planes.len());
    for (pi, _) in planes.iter().enumerate() {
        let inliers = std::mem::take(&mut extended[pi]);
        if inliers.len() < 3 {
            continue;
        }
        let pts3d: Vec<[f32; 3]> = inliers.iter().map(|&i| all_pts[i]).collect();
        let idxs: Vec<usize> = (0..pts3d.len()).collect();
        let (normal, _) = match pca_normal_and_curvature(&pts3d, &idxs) {
            Some(r) => r,
            None => continue,
        };
        let n_f = pts3d.len() as f32;
        let cx = pts3d.iter().map(|p| p[0]).sum::<f32>() / n_f;
        let cy = pts3d.iter().map(|p| p[1]).sum::<f32>() / n_f;
        let cz = pts3d.iter().map(|p| p[2]).sum::<f32>() / n_f;
        let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
        result.push((normal, d, inliers));
    }

    result.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_plane_pts(n: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
        let pts: Vec<[f32; 3]> = (0..n)
            .map(|i| [(i % 10) as f32 * 0.1, (i / 10) as f32 * 0.1, 0.0])
            .collect();
        let idx: Vec<usize> = (0..n).collect();
        (pts, idx)
    }

    #[test]
    fn merge_two_same_plane() {
        let (pts, _) = flat_plane_pts(100);
        let planes = vec![
            ([0f32, 0., 1.], 0.0f32, (0..50).collect::<Vec<_>>()),
            ([0f32, 0., 1.], 0.0f32, (50..100).collect::<Vec<_>>()),
        ];
        let merged = merge_planes(&planes, &pts, 5f32.to_radians(), 0.1, 50);
        assert_eq!(
            merged.len(),
            1,
            "two identical planes should merge into one"
        );
        assert_eq!(merged[0].2.len(), 100);
    }

    #[test]
    fn grow_absorbs_nearby() {
        // Two sets of points: a plane at z=0 and a few near-plane leftovers.
        let mut all_pts: Vec<[f32; 3]> = (0..100)
            .map(|i| [(i % 10) as f32 * 0.1, (i / 10) as f32 * 0.1, 0.0])
            .collect();
        all_pts.push([0.5, 0.5, 0.05]); // near-plane leftover
        let n = all_pts.len();

        let planes = vec![([0f32, 0., 1.], 0.0f32, (0..100).collect::<Vec<_>>())];
        let args = GrowArgs {
            dist_thresh: 0.1,
            use_normal: false,
            normal_cos_thresh: 0.866,
            use_curvature: false,
            max_curvature: 0.05,
            use_connectivity: false,
        };
        let grown = grow_planes(&planes, &all_pts, &args);
        assert_eq!(grown.len(), 1);
        assert_eq!(grown[0].2.len(), n, "leftover should be absorbed");
    }
}
