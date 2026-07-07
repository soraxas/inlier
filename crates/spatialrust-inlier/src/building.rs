//! Full building reconstruction pipeline: **align → split storeys → per-floor
//! walls → footprint-based exterior classification**.
//!
//! This is the reusable core shared by the `inspect_building` example and the
//! gallery's "Building" method, so both run the exact same (validated) logic.
//!
//! ```text
//! raw pts ─► voxel downsample (deterministic, first-insertion order)
//!         ─► per-point normals ─► gravity/Manhattan frame + refine up
//!         ─► align (frame coords, for storey + footprint math only)
//!         ─► find_storeys ─► columnwise 2.5-D labels ─► kNN smoothing
//!         ─► per storey: ManhattanPlanes ─► keep walls (⟂up) ─► merge
//!                      ─► slab-expand for coverage
//!                      ─► footprint flood-fill exterior classification
//!         ─► BuildingScene (original-coord points + walls, ready to render)
//! ```
//!
//! All returned geometry (points, wall normals/`d`, `up`) is in the **original**
//! cloud coordinate frame; the aligned coords are a byproduct exposed only for
//! visualisation. Wall inlier indices point into [`BuildingScene::points`] (the
//! downsampled cloud actually used), not the raw input.

use crate::auto_tune::auto_tune_settings;
use crate::plane_estimation::{
    assign_storeys_columnwise, build_footprint2d, compute_normals, estimate_frame_from_normals,
    find_storeys, refine_up_from_normals, smooth_storey_labels, ManhattanPlanes, PlaneEstimator,
};
use crate::plane_ops::merge_planes;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt().max(1e-9);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Tunable knobs for [`reconstruct_building`]. The RANSAC/merge thresholds are
/// derived per-cloud from [`auto_tune_settings`], so only structural knobs live
/// here.
#[derive(Debug, Clone)]
pub struct BuildingParams {
    /// Voxel size for the working downsample, as `bbox_diag / voxel_div`.
    pub voxel_div: f32,
    /// k for normal estimation.
    pub k: usize,
    /// Minimum density-drop prominence to split a storey (see [`find_storeys`]).
    pub min_prominence: f32,
    /// Footprint morphological-closing iterations (0 = off; raise for clean clouds
    /// where gap-leaks dominate over scan-noise).
    pub fp_close_iters: usize,
}

impl Default for BuildingParams {
    fn default() -> Self {
        Self { voxel_div: 700.0, k: 20, min_prominence: 0.25, fp_close_iters: 0 }
    }
}

/// One exterior/interior-classified wall of a storey.
#[derive(Debug, Clone)]
pub struct BuildingWall {
    /// Wall normal in the original frame (horizontal; sign is arbitrary — the
    /// consumer canonicalises it inward for the dollhouse test).
    pub normal: [f32; 3],
    /// Plane offset: `normal · p + d ≈ 0` on the wall.
    pub d: f32,
    /// Indices into [`BuildingScene::points`].
    pub inlier_indices: Vec<usize>,
    /// `true` if the wall lies on the storey's outer footprint boundary.
    pub is_exterior: bool,
    /// Which storey (0 = ground) this wall belongs to.
    pub storey: usize,
}

/// Output of [`reconstruct_building`].
#[derive(Debug, Clone)]
pub struct BuildingScene {
    /// The downsampled cloud actually processed (original coordinates). Wall
    /// indices and the render target both refer to this.
    pub points: Vec<[f32; 3]>,
    /// `points` expressed in the aligned frame (x=h1, y=h2, z=up). Byproduct for
    /// visualisation; consumers rendering in world space should ignore it.
    pub aligned: Vec<[f32; 3]>,
    /// Per-point storey label (index into `0..n_storeys`).
    pub storey_labels: Vec<usize>,
    /// Classified walls across all storeys.
    pub walls: Vec<BuildingWall>,
    /// Estimated gravity/up direction in the original frame.
    pub up: [f32; 3],
    /// Number of detected storeys (≥1).
    pub n_storeys: usize,
}

/// Deterministic voxel downsample preserving first-insertion (scan) order —
/// randomised `HashMap` order would ripple through kNN ties / clustering / merges
/// and make the whole pipeline non-reproducible.
pub fn voxel_downsample(pts: &[[f32; 3]], voxel: f32) -> Vec<[f32; 3]> {
    use std::collections::HashMap;
    let voxel = voxel.max(1e-6);
    let key = |p: &[f32; 3]| {
        ((p[0] / voxel).floor() as i64, (p[1] / voxel).floor() as i64, (p[2] / voxel).floor() as i64)
    };
    let mut slot: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut cells: Vec<([f64; 3], u32)> = Vec::new();
    for p in pts {
        let idx = *slot.entry(key(p)).or_insert_with(|| {
            cells.push(([0.0; 3], 0));
            cells.len() - 1
        });
        for k in 0..3 {
            cells[idx].0[k] += p[k] as f64;
        }
        cells[idx].1 += 1;
    }
    cells
        .into_iter()
        .map(|(s, n)| {
            [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32]
        })
        .collect()
}

/// Run the full building pipeline on a raw point cloud.
///
/// Never panics; returns an empty-walls scene for degenerate input.
pub fn reconstruct_building(raw: &[[f32; 3]], params: &BuildingParams) -> BuildingScene {
    // Bounding box + diagonal (the scale unit for all relative thresholds).
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in raw {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag =
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let pts = voxel_downsample(raw, diag / params.voxel_div.max(1.0));
    let empty = BuildingScene {
        aligned: pts.clone(),
        storey_labels: vec![0; pts.len()],
        walls: vec![],
        up: [0.0, 0.0, 1.0],
        n_storeys: 0,
        points: pts.clone(),
    };
    if pts.len() < 8 || diag <= 0.0 {
        return empty;
    }

    // Align: gravity + Manhattan frame. up = consensus normal of horizontal
    // surfaces (floors/ceilings) so the floors come out flat.
    let normals = compute_normals(&pts, params.k);
    let frame = estimate_frame_from_normals(&normals);
    let up = refine_up_from_normals(&normals, frame.up);
    let h1 = unit([
        frame.h1[0] - up[0] * dot(frame.h1, up),
        frame.h1[1] - up[1] * dot(frame.h1, up),
        frame.h1[2] - up[2] * dot(frame.h1, up),
    ]);
    let h2 = unit(cross(up, h1));
    let c = {
        let mut s = [0.0f32; 3];
        for p in &pts {
            for k in 0..3 {
                s[k] += p[k];
            }
        }
        [s[0] / pts.len() as f32, s[1] / pts.len() as f32, s[2] / pts.len() as f32]
    };
    let aligned: Vec<[f32; 3]> = pts
        .iter()
        .map(|p| {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            [dot(d, h1), dot(d, h2), dot(d, up)]
        })
        .collect();

    // Storeys: density-drop split → per-column 2.5-D labels → kNN smoothing.
    let storeys = find_storeys(&pts, up, params.min_prominence, diag * 0.05);
    let labels0 = assign_storeys_columnwise(&pts, up, h1, h2, &storeys, diag * 0.03, 0.15);
    let storey_labels = smooth_storey_labels(&pts, &labels0, 16, 2);
    let n_storeys = storeys.len().max(1);
    let mut per_storey: Vec<Vec<usize>> = vec![Vec::new(); n_storeys];
    for (i, &s) in storey_labels.iter().enumerate() {
        per_storey[s].push(i);
    }

    // Per-cloud, noise/density-adaptive thresholds.
    let tuned = auto_tune_settings(&pts);
    let expand = diag * 0.01; // slab half-thickness for coverage fill
    let fp_cell = diag * 0.015; // footprint occupancy cell

    let mut walls: Vec<BuildingWall> = Vec::new();
    for (si, storey_idx) in per_storey.iter().enumerate() {
        let sub: Vec<[f32; 3]> = storey_idx.iter().map(|&i| pts[i]).collect();
        let planes = ManhattanPlanes {
            k: params.k,
            dist_thresh: tuned.dist_thresh,
            angle_thresh: tuned.angle_thresh.max(30.0).to_radians(),
            min_support: tuned.min_cluster_size,
        }
        .estimate(&sub);
        // Keep walls (normal ⟂ up), remap local → global indices.
        let raw_walls: Vec<([f32; 3], f32, Vec<usize>)> = planes
            .into_iter()
            .filter(|(n, _, _)| dot(*n, up).abs() < 0.5)
            .map(|(n, d, local)| (n, d, local.iter().map(|&li| storey_idx[li]).collect()))
            .collect();
        let merged = merge_planes(
            &raw_walls,
            &pts,
            tuned.merge_angle_thresh.to_radians(),
            tuned.merge_dist_thresh,
            tuned.merge_min_pts,
        );

        // Footprint of this storey (aligned x=h1, y=h2) + exterior flood-fill.
        let fp_pts: Vec<(f32, f32)> =
            storey_idx.iter().map(|&g| (aligned[g][0], aligned[g][1])).collect();
        let fp = build_footprint2d(&fp_pts, fp_cell, params.fp_close_iters);

        for (n, d, gidx0) in merged {
            // Wall's in-plane extent from its confident merged inliers.
            let t = unit(cross(up, n));
            let (mut tlo, mut thi) = (f32::MAX, f32::MIN);
            for &g in &gidx0 {
                let tp = dot(pts[g], t);
                tlo = tlo.min(tp);
                thi = thi.max(tp);
            }
            // Slab-expand: absorb storey points within `expand` of the plane and
            // inside its footprint span, so the wall renders full.
            let gidx: Vec<usize> = storey_idx
                .iter()
                .copied()
                .filter(|&g| {
                    (dot(n, pts[g]) + d).abs() < expand && {
                        let tp = dot(pts[g], t);
                        tp >= tlo - expand && tp <= thi + expand
                    }
                })
                .collect();
            // Exterior via footprint boundary (project wall pts + normal to 2-D).
            let wall2d: Vec<(f32, f32)> =
                gidx.iter().map(|&g| (aligned[g][0], aligned[g][1])).collect();
            let (a, b) = (dot(n, h1), dot(n, h2));
            let m = (a * a + b * b).sqrt().max(1e-9);
            let is_exterior = fp.wall_is_exterior(&wall2d, (a / m, b / m));
            walls.push(BuildingWall { normal: n, d, inlier_indices: gidx, is_exterior, storey: si });
        }
    }

    BuildingScene { points: pts, aligned, storey_labels, walls, up, n_storeys }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-storey box room: dominant floor+ceiling (so up resolves
    /// vertical) plus four walls. The pipeline should align, find one storey,
    /// and return vertical walls — the invariants the gallery relies on.
    #[test]
    fn reconstructs_a_simple_box_room() {
        let mut pts = Vec::new();
        // Floor (z=0) and ceiling (z=1.5) as fine 2×2 m grids so horizontal
        // surfaces dominate the normal vote → up ≈ vertical.
        let mut x = 0.0;
        while x <= 2.0 {
            let mut y = 0.0;
            while y <= 2.0 {
                pts.push([x, y, 0.0]);
                pts.push([x, y, 1.5]);
                y += 0.05;
            }
            x += 0.05;
        }
        // Four walls at the box perimeter (coarser).
        let mut a = 0.0;
        while a <= 2.0 {
            let mut z = 0.0;
            while z <= 1.5 {
                pts.push([0.0, a, z]);
                pts.push([2.0, a, z]);
                pts.push([a, 0.0, z]);
                pts.push([a, 2.0, z]);
                z += 0.1;
            }
            a += 0.1;
        }
        let scene = reconstruct_building(&pts, &BuildingParams::default());
        assert!(!scene.points.is_empty());
        assert_eq!(scene.aligned.len(), scene.points.len());
        assert_eq!(scene.storey_labels.len(), scene.points.len());
        assert!(scene.n_storeys >= 1);
        // up should be near vertical (±z).
        assert!(scene.up[2].abs() > 0.8, "up not vertical: {:?}", scene.up);
        // Walls, if any, must be vertical and index in range.
        for w in &scene.walls {
            assert!(dot(w.normal, scene.up).abs() < 0.6, "wall not vertical");
            for &i in &w.inlier_indices {
                assert!(i < scene.points.len());
            }
        }
    }
}
