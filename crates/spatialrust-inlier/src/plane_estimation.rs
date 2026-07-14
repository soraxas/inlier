//! Pluggable plane-estimation methods.
//!
//! A [`PlaneEstimator`] turns a point cloud into a set of planar segments. Each
//! strategy lives behind this one interface so callers (and the gallery UI) can
//! swap methods and A/B them on the same cloud:
//!
//! - [`RegionGrowing`] — Ling et al. 2024 region-growing + RANSAC. Great for
//!   dense, uniformly sampled clouds; fragments on holey / variable-density
//!   scans (e.g. deep-learning point clouds) because growing relies on local
//!   connectivity, which breaks across gaps.
//! - [`GlobalPlanePeeling`] — global dominant-plane extraction with a
//!   normal-consensus filter. Ignores connectivity, so disjoint / holey wall
//!   points still land in one plane; gives high coverage and few large planes,
//!   which is what downstream classification (e.g. dollhouse exterior/interior)
//!   needs.

use crate::normals::pca_normal_and_curvature;
use crate::plane::fit_plane_msac;
use crate::region_growing::{RansacMode, region_growing_ransac_with_progress};
use crate::spatial_grid::{build_grid, estimate_cell_size, knn};

/// One planar segment: `(unit_normal, plane_offset_d, inlier_indices)`, where a
/// point `p` on the plane satisfies `normal · p + d ≈ 0`. Indices refer to the
/// input point slice. This matches the tuple used throughout the crate.
pub type Plane = ([f32; 3], f32, Vec<usize>);

/// Common interface for plane-estimation strategies.
pub trait PlaneEstimator {
    /// Estimate planar segments, reporting `(fraction ∈ [0,1], phase_label)`
    /// through `on_progress` so callers can drive a progress bar while the
    /// estimate runs off the main thread.
    fn estimate_with_progress(
        &self,
        pts: &[[f32; 3]],
        on_progress: &mut dyn FnMut(f32, &str),
    ) -> Vec<Plane>;

    /// Estimate planar segments without progress reporting.
    fn estimate(&self, pts: &[[f32; 3]]) -> Vec<Plane> {
        self.estimate_with_progress(pts, &mut |_, _| {})
    }
}

/// Region-growing + RANSAC plane segmentation (Ling et al. 2024).
///
/// Estimates per-point normals/curvature, grows curvature-seeded regions by
/// local normal agreement, then fits a plane per region. See
/// [`region_growing_ransac`](crate::region_growing::region_growing_ransac).
#[derive(Debug, Clone)]
pub struct RegionGrowing {
    /// Neighbourhood size for normal/curvature estimation.
    pub k: usize,
    /// Max angle (radians) between neighbour normals to keep growing a region.
    pub angle_thresh: f32,
    /// Discard regions/planes with fewer inliers than this.
    pub min_cluster_size: usize,
    /// Point-to-plane inlier distance threshold.
    pub dist_thresh: f32,
    /// Per-region plane fitter (Simple / MSAC / MAGSAC++).
    pub mode: RansacMode,
    /// MAGSAC++ σ_max (ignored for Simple/MSAC).
    pub sigma_max: f64,
    /// MSAC/MAGSAC++ iteration budget.
    pub max_iterations: usize,
    /// MSAC/MAGSAC++ confidence.
    pub confidence: f64,
}

impl PlaneEstimator for RegionGrowing {
    fn estimate_with_progress(
        &self,
        pts: &[[f32; 3]],
        on_progress: &mut dyn FnMut(f32, &str),
    ) -> Vec<Plane> {
        region_growing_ransac_with_progress(
            pts,
            self.k,
            self.angle_thresh,
            self.min_cluster_size,
            self.dist_thresh,
            self.mode,
            self.sigma_max,
            self.max_iterations,
            self.confidence,
            on_progress,
        )
    }
}

/// Global dominant-plane peeling with a normal-consensus filter.
///
/// Repeatedly fits the single most-supported plane over *all* remaining points
/// (MSAC), keeps as inliers only those points whose own normal agrees with the
/// plane normal (rejecting points from transverse surfaces that merely fall in
/// the slab), removes them, and repeats. Because the fit is global, a wall
/// broken into disjoint pieces by missing data still becomes one plane — unlike
/// region growing. Produces few large planes with high coverage.
#[derive(Debug, Clone)]
pub struct GlobalPlanePeeling {
    /// Neighbourhood size for per-point normal estimation.
    pub k: usize,
    /// Point-to-plane inlier distance threshold.
    pub dist_thresh: f32,
    /// If true, exclude points whose (valid) normal is more than `angle_thresh`
    /// from the plane normal — an anti-cutting filter. Costs a per-point normal
    /// pass and, on clouds with unreliable normals (variable density / noise),
    /// hurts coverage. Default to `false`: membership is then purely by
    /// distance, and MSAC's dominant-plane selection avoids cutting planes.
    pub normal_consensus: bool,
    /// Max angle (radians) from the plane normal, used only when
    /// `normal_consensus` is true.
    pub angle_thresh: f32,
    /// Stop peeling once the best plane has fewer than this many inliers (also
    /// the minimum to keep a plane).
    pub min_support: usize,
    /// Hard cap on the number of planes.
    pub max_planes: usize,
    /// MSAC iteration budget per plane fit.
    pub max_iterations: usize,
    /// MSAC confidence.
    pub confidence: f64,
}

impl PlaneEstimator for GlobalPlanePeeling {
    fn estimate_with_progress(
        &self,
        pts: &[[f32; 3]],
        on_progress: &mut dyn FnMut(f32, &str),
    ) -> Vec<Plane> {
        let n = pts.len();
        if n < 3 {
            return vec![];
        }

        // Per-point normals only when the anti-cutting gate is enabled.
        let normals: Vec<[f32; 3]> = if self.normal_consensus {
            on_progress(0.0, "Estimating normals");
            let cell = estimate_cell_size(pts);
            let grid = build_grid(pts, cell);
            (0..n)
                .map(|i| {
                    let nb = knn(pts, i, self.k, cell, &grid);
                    if nb.len() < 3 {
                        [0.0; 3]
                    } else {
                        pca_normal_and_curvature(pts, &nb)
                            .map(|(nv, _)| nv)
                            .unwrap_or([0.0; 3])
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        on_progress(0.2, "Peeling planes");

        let cos_t = self.angle_thresh.cos();
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut planes: Vec<Plane> = Vec::new();

        while planes.len() < self.max_planes && remaining.len() >= self.min_support {
            // Fit the dominant plane over ALL remaining points.
            let sub: Vec<[f32; 3]> = remaining.iter().map(|&i| pts[i]).collect();
            let settings = Some(inlier::MetasacSettings {
                max_iterations: self.max_iterations,
                confidence: self.confidence,
                ..inlier::MetasacSettings::default()
            });
            let Some((nrm, d)) =
                fit_plane_msac(&sub, self.dist_thresh as f64, settings).map(|(nv, dv, _)| (nv, dv))
            else {
                break;
            };

            // Membership is distance-based (so points with bad/missing normals in
            // sparse regions still get assigned → high coverage). The normal gate
            // only *excludes* points whose normal is valid AND confidently
            // transverse to the plane — i.e. they belong to a crossing surface,
            // not this one. This keeps the anti-cutting behaviour where normals
            // are reliable without sacrificing coverage where they aren't.
            let (mut inliers, mut keep) = (Vec::new(), Vec::new());
            for &i in &remaining {
                let p = pts[i];
                let dist = (nrm[0] * p[0] + nrm[1] * p[1] + nrm[2] * p[2] + d).abs();
                // Anti-cutting gate: reject only points with a valid normal that
                // is confidently transverse to the plane. Off by default.
                let transverse = if self.normal_consensus {
                    let nv = normals[i];
                    let dot = (nv[0] * nrm[0] + nv[1] * nrm[1] + nv[2] * nrm[2]).abs();
                    nv != [0.0; 3] && dot < cos_t
                } else {
                    false
                };
                if dist < self.dist_thresh && !transverse {
                    inliers.push(i);
                } else {
                    keep.push(i);
                }
            }

            // If the dominant plane no longer has real support, stop.
            if inliers.len() < self.min_support {
                break;
            }
            planes.push((nrm, d, inliers));
            remaining = keep;
            on_progress(
                0.2 + 0.8 * (1.0 - remaining.len() as f32 / n as f32),
                "Peeling planes",
            );
        }

        on_progress(1.0, "Done");
        planes.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
        planes
    }
}

/// Manhattan-frame plane extraction: bias fits to a shared orthogonal frame so
/// planes come out in *all* orientations (walls in two directions + floors),
/// instead of greedy peeling repeatedly taking the largest-area orientation.
///
/// 1. Estimate an orthogonal frame from normal statistics: `up` = the dominant
///    normal direction (floors/ceilings have the most points); `h1` = the
///    dominant *horizontal* normal (main wall direction); `h2 = up × h1`.
/// 2. Assign each point to the frame axis its normal is closest to (an
///    orientation vote), so a horizontal slab can't absorb wall points.
/// 3. Along each axis, gap-cluster the assigned points' projections: each tight
///    band is one plane (a floor at an `up` offset, a wall at an `h1`/`h2`
///    offset). Plane normals are snapped exactly to the frame axis.
///
/// Assumes walls are near-orthogonal (Manhattan world) — true for buildings.
/// Slanted/curved surfaces fall between axes and stay unassigned.
#[derive(Debug, Clone)]
pub struct ManhattanPlanes {
    /// Neighbourhood size for per-point normal estimation.
    pub k: usize,
    /// Slab half-width: gap between projection bands that splits two planes,
    /// and the tolerance that binds a band together.
    pub dist_thresh: f32,
    /// Max angle (radians) between a point normal and a frame axis for the point
    /// to be assigned to that axis.
    pub angle_thresh: f32,
    /// Minimum points for a band to become a plane.
    pub min_support: usize,
}

/// Dominant direction of a set of normals via the largest eigenvector of
/// `Σ vvᵀ`. With `up=None`, `v = n` → the most common normal (≈ up). With
/// `up=Some`, `v` is the component of `n` orthogonal to `up` → the dominant
/// An orthonormal building frame: `up` (gravity, dominant normal), `h1`/`h2`
/// the two horizontal wall directions. Project a point onto an axis with a dot
/// product; no need to materialize a rotated cloud.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub up: [f32; 3],
    pub h1: [f32; 3],
    pub h2: [f32; 3],
}

/// Per-point normals via kNN PCA (unit; `[0;3]` where a point has < 3 neighbours).
pub fn compute_normals(pts: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    let cell = estimate_cell_size(pts);
    let grid = build_grid(pts, cell);
    (0..pts.len())
        .map(|i| {
            let nb = knn(pts, i, k, cell, &grid);
            if nb.len() < 3 {
                [0.0; 3]
            } else {
                pca_normal_and_curvature(pts, &nb)
                    .map(|(nv, _)| nv)
                    .unwrap_or([0.0; 3])
            }
        })
        .collect()
}

/// Estimate the orthogonal building frame from point normals: `up` = dominant
/// normal (floors/ceilings dominate); `h1` = dominant horizontal wall direction
/// (mod-90° angle histogram); `h2 = up × h1`.
pub fn estimate_frame_from_normals(normals: &[[f32; 3]]) -> Frame {
    let up = dominant_direction(normals, None);
    let h1 = dominant_horizontal(normals, up);
    let mut h2 = cross3(up, h1);
    let n = (h2[0] * h2[0] + h2[1] * h2[1] + h2[2] * h2[2]).sqrt();
    if n > 1e-6 {
        h2 = [h2[0] / n, h2[1] / n, h2[2] / n];
    }
    Frame { up, h1, h2 }
}

/// Estimate the building frame directly from a point cloud (`k` = normal kNN).
pub fn estimate_frame(pts: &[[f32; 3]], k: usize) -> Frame {
    estimate_frame_from_normals(&compute_normals(pts, k))
}

/// Refine up to true gravity as the *consensus normal of horizontal surfaces*.
/// Iteratively: keep normals within `cone` of the current up (floors/ceilings,
/// excluding walls), set up to their dominant direction (largest eigenvector of
/// Σ nnᵀ), repeat. Converges to the floor/ceiling normal — the direction along
/// which floors are truly flat — which histogram-sharpness search can miss.
pub fn refine_up_from_normals(normals: &[[f32; 3]], up0: [f32; 3]) -> [f32; 3] {
    use nalgebra::{Matrix3, SymmetricEigen, Vector3};
    let cone = 25f32.to_radians().cos();
    let mut up = up0;
    for _ in 0..6 {
        let mut m = Matrix3::<f64>::zeros();
        for &nv in normals {
            if nv == [0.0; 3] {
                continue;
            }
            let v = Vector3::new(nv[0] as f64, nv[1] as f64, nv[2] as f64).normalize();
            let dot = (v[0] as f32 * up[0] + v[1] as f32 * up[1] + v[2] as f32 * up[2]).abs();
            if dot < cone {
                continue; // skip wall (transverse) normals
            }
            m += v * v.transpose();
        }
        if m.trace() < 1e-9 {
            break;
        }
        let eig = SymmetricEigen::new(m);
        let mut best = 0;
        for i in 1..3 {
            if eig.eigenvalues[i] > eig.eigenvalues[best] {
                best = i;
            }
        }
        let e = eig.eigenvectors.column(best);
        let mut new_up = Vector3::new(e[0], e[1], e[2]).normalize();
        // Keep the original hemisphere.
        if new_up.dot(&Vector3::new(up0[0] as f64, up0[1] as f64, up0[2] as f64)) < 0.0 {
            new_up = -new_up;
        }
        let nu = [new_up[0] as f32, new_up[1] as f32, new_up[2] as f32];
        let change = 1.0 - (nu[0] * up[0] + nu[1] * up[1] + nu[2] * up[2]).abs();
        up = nu;
        if change < 1e-5 {
            break;
        }
    }
    up
}

/// Refine an up estimate to true gravity by maximizing height-histogram
/// *sharpness* (Σ count²): the direction along which floors/ceilings collapse
/// to the tallest peaks is gravity. Robust to noisy per-point normals — a small
/// tilt in the normal-based up smears the histogram, and this corrects it.
/// Searches a ±12° grid around `up0`.
pub fn refine_up(pts: &[[f32; 3]], up0: [f32; 3]) -> [f32; 3] {
    if pts.len() < 3 {
        return up0;
    }
    let a = if up0[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut t1 = cross3(up0, a);
    let n1 = (t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]).sqrt();
    t1 = [t1[0] / n1, t1[1] / n1, t1[2] / n1];
    let t2 = cross3(up0, t1); // unit

    let sharpness = |u: [f32; 3]| -> f64 {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for p in pts {
            let h = dot3(*p, u);
            lo = lo.min(h);
            hi = hi.max(h);
        }
        let nb = 200usize;
        let w = (hi - lo) / nb as f32;
        if w < 1e-9 {
            return 0.0;
        }
        let mut c = vec![0.0f64; nb];
        for p in pts {
            let b = (((dot3(*p, u) - lo) / w) as usize).min(nb - 1);
            c[b] += 1.0;
        }
        c.iter().map(|x| x * x).sum()
    };

    let step = 2f32.to_radians();
    let (mut best_u, mut best_s) = (up0, sharpness(up0));
    for i in -6..=6 {
        for j in -6..=6 {
            let (a1, a2) = (i as f32 * step, j as f32 * step);
            let mut u = [
                up0[0] + t1[0] * a1 + t2[0] * a2,
                up0[1] + t1[1] * a1 + t2[1] * a2,
                up0[2] + t1[2] * a1 + t2[2] * a2,
            ];
            let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            u = [u[0] / un, u[1] / un, u[2] / un];
            let s = sharpness(u);
            if s > best_s {
                best_s = s;
                best_u = u;
            }
        }
    }
    best_u
}

/// Split a cloud into storeys (vertical levels) via a density histogram along
/// `up`. Floors/ceilings are high-density peaks; a *prominent valley* between
/// them (empty space between one level's ceiling and the next level's floor)
/// separates levels. Returns each level's `(min_height, max_height)` along `up`.
///
/// `min_prominence` (0..1): a valley splits levels only if its density drops
/// below this fraction of the smaller flanking peak. `min_height` drops levels
/// thinner than this (in the cloud's units).
pub fn find_storeys(
    pts: &[[f32; 3]],
    up: [f32; 3],
    min_prominence: f32,
    min_height: f32,
) -> Vec<(f32, f32)> {
    let n = pts.len();
    if n == 0 {
        return vec![];
    }
    let hs: Vec<f32> = pts.iter().map(|p| dot3(*p, up)).collect();
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &h in &hs {
        lo = lo.min(h);
        hi = hi.max(h);
    }
    let range = hi - lo;
    if range < min_height {
        return vec![(lo, hi)];
    }
    let nb = 200usize;
    let w = range / nb as f32;
    let mut counts = vec![0.0f32; nb];
    for &h in &hs {
        let b = (((h - lo) / w) as usize).min(nb - 1);
        counts[b] += 1.0;
    }
    // Heavy smoothing (window ≈ min_height): merges each level's internal
    // structure (floor / sparse interior / ceiling) into ONE hump, so only
    // level-to-level transitions survive. Without this, the floor→interior drop
    // inside a single room would look like a level boundary.
    let wh = ((min_height / w) as usize).clamp(3, nb / 2);
    let mut hs_sm = vec![0.0f32; nb];
    for i in 0..nb {
        let a = i.saturating_sub(wh / 2);
        let b = (i + wh / 2 + 1).min(nb);
        hs_sm[i] = counts[a..b].iter().sum::<f32>() / (b - a).max(1) as f32;
    }
    let peak = hs_sm.iter().cloned().fold(0.0f32, f32::max).max(1.0);

    // Level boundaries are where density DROPS going up — leaving a dense level
    // into an empty gap OR stepping down to a sparser (smaller-footprint) level.
    // A drop handles both the "valley" and the "step" case; within-level
    // structure has no drop after heavy smoothing. Entry/exit near the top and
    // bottom extremes are excluded.
    let g = (wh / 2).max(1);
    let edge = ((min_height / w) as usize).max(1);
    let mut drop = vec![0.0f32; nb];
    for i in g..nb - g {
        drop[i] = hs_sm[i - g] - hs_sm[i + g];
    }
    let mut cand: Vec<(f32, f32)> = Vec::new(); // (height, drop magnitude)
    for i in edge.max(1)..nb.saturating_sub(edge).min(nb - 1) {
        if drop[i] > min_prominence * peak && drop[i] >= drop[i - 1] && drop[i] >= drop[i + 1] {
            cand.push((lo + (i as f32 + 0.5) * w, drop[i]));
        }
    }
    // Greedily accept the steepest drops as splits, requiring every resulting
    // level to be ≥ min_height (each split ≥ min_height from the extremes and
    // from already-accepted splits). This guarantees valid levels by
    // construction — no band is ever dropped.
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut splits: Vec<f32> = Vec::new();
    for (h, _) in cand {
        if h - lo >= min_height
            && hi - h >= min_height
            && splits.iter().all(|&s| (s - h).abs() >= min_height)
        {
            splits.push(h);
        }
    }
    splits.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut bounds = vec![lo];
    bounds.extend(splits);
    bounds.push(hi);
    bounds.windows(2).map(|b| (b[0], b[1])).collect()
}

/// Per-column (2.5-D) storey assignment. A global height split mislabels the
/// region with no upper level — there, the ground *ceiling* falls in the upper
/// band and reads as "upper." Fix: bin the footprint (h1,h2) into cells and, in
/// each cell, only keep a storey band if it is genuinely occupied there
/// (≥ `min_occ_frac` of the cell's points). Each point is then capped to the
/// highest occupied band at or below its global band — so a ground-only column
/// stays all-ground, and the upper storey appears only under the upper room.
/// Returns a storey index per point (0 = lowest).
pub fn assign_storeys_columnwise(
    pts: &[[f32; 3]],
    up: [f32; 3],
    h1: [f32; 3],
    h2: [f32; 3],
    storeys: &[(f32, f32)],
    cell: f32,
    min_occ_frac: f32,
) -> Vec<usize> {
    use std::collections::HashMap;
    let nb = storeys.len().max(1);
    if storeys.len() <= 1 {
        return vec![0; pts.len()];
    }
    let band_of = |h: f32| -> usize {
        for (i, &(a, b)) in storeys.iter().enumerate() {
            if h >= a && h < b {
                return i;
            }
        }
        if h < storeys[0].0 { 0 } else { nb - 1 }
    };
    let cell = cell.max(1e-6);
    let key = |p: &[f32; 3]| {
        (
            (dot3(*p, h1) / cell).floor() as i64,
            (dot3(*p, h2) / cell).floor() as i64,
        )
    };
    let bands: Vec<usize> = pts.iter().map(|p| band_of(dot3(*p, up))).collect();

    // Per-cell band counts.
    let mut counts: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for (p, &bd) in pts.iter().zip(&bands) {
        counts.entry(key(p)).or_insert_with(|| vec![0; nb])[bd] += 1;
    }
    // Per-cell occupancy mask (≥ frac of the cell's points).
    let occ: HashMap<(i64, i64), Vec<bool>> = counts
        .iter()
        .map(|(k, v)| {
            let total: u32 = v.iter().sum();
            let the = (min_occ_frac * total as f32).max(1.0);
            (*k, v.iter().map(|&c| c as f32 >= the).collect())
        })
        .collect();

    // Cap each point to the highest occupied band ≤ its band.
    pts.iter()
        .zip(&bands)
        .map(|(p, &bd)| {
            let o = &occ[&key(p)];
            (0..=bd).rev().find(|&i| o[i]).unwrap_or(0)
        })
        .collect()
}

/// Smooth storey labels by spatial majority vote (a region-growing-style pass):
/// each point takes the most common label among its k nearest neighbours,
/// iterated `iters` times. Flips isolated boundary points to match their
/// neighbourhood — reducing salt-and-pepper storey bleed at the seam — while a
/// small `k` preserves the true level boundary.
pub fn smooth_storey_labels(
    pts: &[[f32; 3]],
    labels: &[usize],
    k: usize,
    iters: usize,
) -> Vec<usize> {
    use std::collections::HashMap;
    let cell = estimate_cell_size(pts);
    let grid = build_grid(pts, cell);
    let mut cur = labels.to_vec();
    for _ in 0..iters {
        let next: Vec<usize> = (0..pts.len())
            .map(|i| {
                let nb = knn(pts, i, k, cell, &grid);
                let mut counts: HashMap<usize, u32> = HashMap::new();
                *counts.entry(cur[i]).or_insert(0) += 1;
                for &j in &nb {
                    *counts.entry(cur[j]).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(l, _)| l)
                    .unwrap_or(cur[i])
            })
            .collect();
        cur = next;
    }
    cur
}

/// A storey's 2-D footprint occupancy grid, plus the border-reachable *exterior*
/// empty space (flood-filled from the grid edge through empty cells). Enclosed
/// empty regions (interior rooms, courtyards) are NOT reached, so they read as
/// interior. Used to classify exterior walls robustly on concave footprints.
pub struct Footprint2D {
    ext: Vec<bool>,
    nx: usize,
    ny: usize,
    cell: f32,
    ox: f32,
    oy: f32,
    reach: f32,
}

/// Build a footprint occupancy grid from 2-D points and flood-fill the exterior.
///
/// `close_iters` dilates the occupancy by that many cells before flooding
/// (morphological closing) — this seals doorway/scan gaps up to `2·close_iters`
/// cells wide so the exterior flood can't leak into the interior through holes
/// in a holey (deep-learning) cloud. Enclosed regions (interior rooms, real
/// courtyards) stay unflooded and therefore read as interior.
pub fn build_footprint2d(pts: &[(f32, f32)], cell: f32, close_iters: usize) -> Footprint2D {
    let cell = cell.max(1e-6);
    let (mut minx, mut miny) = (f32::MAX, f32::MAX);
    let (mut maxx, mut maxy) = (f32::MIN, f32::MIN);
    for &(x, y) in pts {
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    // Pad the border by the dilation width + 1 so the flood always has a free
    // border ring to seed from.
    let pad = close_iters + 1;
    let (ox, oy) = (minx - cell * pad as f32, miny - cell * pad as f32);
    let nx = ((maxx - minx) / cell).ceil() as usize + 2 * pad + 1;
    let ny = ((maxy - miny) / cell).ceil() as usize + 2 * pad + 1;
    let at = |i: usize, j: usize| j * nx + i;
    let mut occ = vec![false; nx * ny];
    for &(x, y) in pts {
        let i = ((x - ox) / cell) as usize;
        let j = ((y - oy) / cell) as usize;
        if i < nx && j < ny {
            occ[at(i, j)] = true;
        }
    }
    // Morphological dilation: seal gaps so the flood can't leak through holes.
    for _ in 0..close_iters {
        let prev = occ.clone();
        for j in 0..ny {
            for i in 0..nx {
                if prev[at(i, j)] {
                    continue;
                }
                let hit = (i > 0 && prev[at(i - 1, j)])
                    || (i + 1 < nx && prev[at(i + 1, j)])
                    || (j > 0 && prev[at(i, j - 1)])
                    || (j + 1 < ny && prev[at(i, j + 1)]);
                if hit {
                    occ[at(i, j)] = true;
                }
            }
        }
    }
    // Flood empty cells from the border (4-connectivity).
    let mut ext = vec![false; nx * ny];
    let mut q = std::collections::VecDeque::new();
    let mut seed = |i: usize,
                    j: usize,
                    ext: &mut Vec<bool>,
                    q: &mut std::collections::VecDeque<(usize, usize)>| {
        if !occ[at(i, j)] && !ext[at(i, j)] {
            ext[at(i, j)] = true;
            q.push_back((i, j));
        }
    };
    for i in 0..nx {
        seed(i, 0, &mut ext, &mut q);
        seed(i, ny - 1, &mut ext, &mut q);
    }
    for j in 0..ny {
        seed(0, j, &mut ext, &mut q);
        seed(nx - 1, j, &mut ext, &mut q);
    }
    while let Some((i, j)) = q.pop_front() {
        for (a, b) in [
            (i.wrapping_sub(1), j),
            (i + 1, j),
            (i, j.wrapping_sub(1)),
            (i, j + 1),
        ] {
            if a < nx && b < ny && !occ[at(a, b)] && !ext[at(a, b)] {
                ext[at(a, b)] = true;
                q.push_back((a, b));
            }
        }
    }
    // A wall probe must clear the dilation collar (close_iters cells) plus a
    // base 3-cell reach to sit in the flooded exterior.
    let reach = cell * (close_iters as f32 + 3.0);
    Footprint2D {
        ext,
        nx,
        ny,
        cell,
        ox,
        oy,
        reach,
    }
}

impl Footprint2D {
    fn is_ext(&self, x: f32, y: f32) -> bool {
        let i = ((x - self.ox) / self.cell).floor() as isize;
        let j = ((y - self.oy) / self.cell).floor() as isize;
        if i < 0 || j < 0 || i as usize >= self.nx || j as usize >= self.ny {
            return true; // beyond the padded grid = outside
        }
        self.ext[j as usize * self.nx + i as usize]
    }

    /// A wall is exterior if a fair fraction of its footprint points sit on the
    /// boundary — exterior empty space on one side, not the other. Interior
    /// partitions have interior/occupied on both sides; courtyard walls border
    /// enclosed (non-exterior) empty space, so they read interior too.
    pub fn wall_is_exterior(&self, wall_pts: &[(f32, f32)], n: (f32, f32)) -> bool {
        if wall_pts.is_empty() {
            return false;
        }
        let mut border = 0usize;
        for &(x, y) in wall_pts {
            let pos = self.is_ext(x + n.0 * self.reach, y + n.1 * self.reach);
            let neg = self.is_ext(x - n.0 * self.reach, y - n.1 * self.reach);
            if pos != neg {
                border += 1;
            }
        }
        border as f32 / wall_pts.len() as f32 > 0.3
    }
}

/// horizontal normal.
fn dominant_direction(normals: &[[f32; 3]], up: Option<[f32; 3]>) -> [f32; 3] {
    use nalgebra::{Matrix3, SymmetricEigen, Vector3};
    let mut m = Matrix3::<f64>::zeros();
    for &nv in normals {
        if nv == [0.0; 3] {
            continue;
        }
        let mut v = Vector3::new(nv[0] as f64, nv[1] as f64, nv[2] as f64);
        if let Some(u) = up {
            let uu = Vector3::new(u[0] as f64, u[1] as f64, u[2] as f64);
            v -= uu * v.dot(&uu);
        }
        let nrm = v.norm();
        if nrm < 1e-6 {
            continue;
        }
        v /= nrm;
        m += v * v.transpose();
    }
    let eig = SymmetricEigen::new(m);
    let mut best = 0;
    for i in 1..3 {
        if eig.eigenvalues[i] > eig.eigenvalues[best] {
            best = i;
        }
    }
    let e = eig.eigenvectors.column(best);
    let v = Vector3::new(e[0], e[1], e[2]).normalize();
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// Dominant *horizontal* wall direction as the peak of a mod-90° orientation
/// histogram (circular mean of 4θ). Unlike an eigenvector, this stays stable
/// when the two perpendicular wall directions have equal support — walls at θ
/// and θ+90° reinforce the *same* Manhattan peak instead of cancelling into a
/// diagonal.
fn dominant_horizontal(normals: &[[f32; 3]], up: [f32; 3]) -> [f32; 3] {
    let a = if up[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut e1 = cross3(up, a);
    let e1n = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
    e1 = [e1[0] / e1n, e1[1] / e1n, e1[2] / e1n];
    let e2 = cross3(up, e1); // unit: up ⟂ e1, both unit
    let (mut c, mut s) = (0.0f64, 0.0f64);
    for &n in normals {
        if n == [0.0; 3] {
            continue;
        }
        let du = dot3(n, up);
        let h = [n[0] - up[0] * du, n[1] - up[1] * du, n[2] - up[2] * du];
        let x = dot3(h, e1) as f64;
        let y = dot3(h, e2) as f64;
        let w = (x * x + y * y).sqrt(); // horizontal magnitude (floors weigh ~0)
        if w < 1e-4 {
            continue;
        }
        let th = y.atan2(x);
        c += w * (4.0 * th).cos();
        s += w * (4.0 * th).sin();
    }
    let phi = (s.atan2(c) / 4.0) as f32;
    let (cp, sp) = (phi.cos(), phi.sin());
    [
        e1[0] * cp + e2[0] * sp,
        e1[1] * cp + e2[1] * sp,
        e1[2] * cp + e2[2] * sp,
    ]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cluster 1-D projections into planes by density peaks. Histograms the values
/// (bin = `dist`), takes local-maximum bins with ≥ `min_support` as plane
/// centers (merging centers closer than `dist`), then assigns each value to its
/// nearest center within `dist`. Returns `(center, member_indices)` per plane.
/// Robust to sparse stray points that would chain single-linkage clusters.
fn cluster_1d_peaks(
    mut items: Vec<(f32, usize)>,
    dist: f32,
    min_support: usize,
) -> Vec<(f32, Vec<usize>)> {
    if items.len() < min_support {
        return vec![];
    }
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (lo, hi) = (items[0].0, items[items.len() - 1].0);
    let w = dist.max(1e-6);
    let nb = ((hi - lo) / w) as usize + 1;
    let mut counts = vec![0usize; nb];
    for &(p, _) in &items {
        counts[((p - lo) / w) as usize] += 1;
    }
    // Local-maximum bins with enough support → candidate centers.
    let mut centers: Vec<f32> = Vec::new();
    for i in 0..nb {
        let c = counts[i];
        if c >= min_support && (i == 0 || c >= counts[i - 1]) && (i + 1 == nb || c >= counts[i + 1])
        {
            let center = lo + (i as f32 + 0.5) * w;
            // Merge with the previous center if within `dist`.
            if centers.last().map(|&p| center - p < dist).unwrap_or(false) {
                continue;
            }
            centers.push(center);
        }
    }
    if centers.is_empty() {
        return vec![];
    }
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];
    let mut sums = vec![0.0f32; centers.len()];
    for &(p, idx) in &items {
        let mut bi = 0;
        let mut bd = f32::MAX;
        for (ci, &c) in centers.iter().enumerate() {
            let d = (p - c).abs();
            if d < bd {
                bd = d;
                bi = ci;
            }
        }
        if bd <= dist {
            buckets[bi].push(idx);
            sums[bi] += p;
        }
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, b)| b.len() >= min_support)
        .map(|(ci, b)| (sums[ci] / b.len() as f32, b))
        .collect()
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl PlaneEstimator for ManhattanPlanes {
    fn estimate_with_progress(
        &self,
        pts: &[[f32; 3]],
        on_progress: &mut dyn FnMut(f32, &str),
    ) -> Vec<Plane> {
        let n = pts.len();
        if n < 3 {
            return vec![];
        }

        // 1. Per-point normals.
        on_progress(0.0, "Estimating normals");
        let normals = compute_normals(pts, self.k);

        // 2. Orthogonal frame from normal statistics.
        on_progress(0.5, "Estimating frame");
        let frame = estimate_frame_from_normals(&normals);
        let axes = [frame.up, frame.h1, frame.h2];

        // 3. Orientation vote: assign each point to the axis its normal matches.
        on_progress(0.6, "Extracting planes");
        let cos_t = self.angle_thresh.cos();
        let mut groups: [Vec<(f32, usize)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..n {
            let nv = normals[i];
            if nv == [0.0; 3] {
                continue;
            }
            let d = [
                dot3(nv, axes[0]).abs(),
                dot3(nv, axes[1]).abs(),
                dot3(nv, axes[2]).abs(),
            ];
            let a = if d[0] >= d[1] && d[0] >= d[2] {
                0
            } else if d[1] >= d[2] {
                1
            } else {
                2
            };
            if d[a] < cos_t {
                continue;
            }
            groups[a].push((dot3(pts[i], axes[a]), i));
        }

        // 4. Cluster projections along each axis into parallel planes by DENSITY
        //    peaks (not single-linkage gaps, which would chain distinct floors
        //    together through sparse stray points).
        let mut planes: Vec<Plane> = Vec::new();
        for a in 0..3 {
            let g = std::mem::take(&mut groups[a]);
            for (center, idx) in cluster_1d_peaks(g, self.dist_thresh, self.min_support) {
                planes.push((axes[a], -center, idx));
            }
        }

        on_progress(1.0, "Done");
        planes.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
        planes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region_growing::region_growing_ransac;

    // Reuse the region_growing synthetic-cloud helper via a tiny local copy of
    // the parameters used by its own smoke test, and assert the trait wrapper
    // matches the free function exactly.
    #[test]
    fn region_growing_estimator_matches_free_function() {
        // Three axis-aligned planes with noise + outliers (same shape as the
        // region_growing smoke test).
        let mut pts = Vec::new();
        let mut s: u64 = 42;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        for _ in 0..600 {
            pts.push([rnd(), rnd(), 0.0 + 0.01 * rnd()]);
        }
        for _ in 0..600 {
            pts.push([rnd(), 0.0 + 0.01 * rnd(), rnd()]);
        }
        for _ in 0..600 {
            pts.push([0.0 + 0.01 * rnd(), rnd(), rnd()]);
        }

        let est = RegionGrowing {
            k: 20,
            angle_thresh: 10f32.to_radians(),
            min_cluster_size: 30,
            dist_thresh: 0.08,
            mode: RansacMode::Simple,
            sigma_max: 0.0,
            max_iterations: 1000,
            confidence: 0.99,
        };

        let via_trait = est.estimate(&pts);
        let via_free = region_growing_ransac(
            &pts,
            est.k,
            est.angle_thresh,
            est.min_cluster_size,
            est.dist_thresh,
            est.mode,
            est.sigma_max,
            est.max_iterations,
            est.confidence,
        );
        assert_eq!(via_trait.len(), via_free.len());
        for (a, b) in via_trait.iter().zip(via_free.iter()) {
            assert_eq!(a.2, b.2);
        }
    }

    #[test]
    fn smooth_flips_isolated_labels() {
        // Dense grid, all label 0 except one interior stray label-1 point.
        let mut pts = Vec::new();
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    pts.push([x as f32, y as f32, z as f32]);
                }
            }
        }
        let mut labels = vec![0usize; pts.len()];
        let stray = pts.iter().position(|p| *p == [4.0, 4.0, 4.0]).unwrap();
        labels[stray] = 1;
        let out = smooth_storey_labels(&pts, &labels, 8, 1);
        assert_eq!(
            out[stray], 0,
            "isolated label should flip to its neighbourhood"
        );
    }

    #[test]
    fn columnwise_storeys_no_leak_into_empty_columns() {
        // Ground level (z∈[0,1]) over the full plan x∈[-2,2]; upper level
        // (z∈[2,3]) only over x<0. A global height split would label any high
        // point on x>0 as "upper"; per-column must cap x>0 to ground.
        let mut pts = Vec::new();
        let mut s: u64 = 5;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        for _ in 0..3000 {
            pts.push([rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0, rnd()]);
        } // ground, all x
        for _ in 0..1500 {
            pts.push([rnd() * 2.0 - 2.0, rnd() * 4.0 - 2.0, 2.0 + rnd()]);
        } // upper, x<0
        // A few stray high points on x>0 (e.g. ceiling grazing) — must NOT be "upper".
        for _ in 0..40 {
            pts.push([rnd() * 2.0, rnd() * 4.0 - 2.0, 2.0 + rnd()]);
        }

        let storeys = [(-0.5f32, 1.5f32), (1.5, 3.5)];
        let labels = assign_storeys_columnwise(
            &pts,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            &storeys,
            0.5,
            0.2,
        );
        // Every "upper" label must sit on x<0 (the real upper-room footprint).
        for (p, &lab) in pts.iter().zip(&labels) {
            if lab == 1 {
                assert!(p[0] < 0.2, "upper label leaked to x>0 at {p:?}");
            }
        }
        // The genuine upper room (x<0, z>2) is labeled upper.
        let upper = labels.iter().filter(|&&l| l == 1).count();
        assert!(
            upper > 1000,
            "expected the x<0 upper room labeled upper, got {upper}"
        );
    }

    #[test]
    fn find_storeys_splits_two_levels() {
        // Two stacked levels separated by an empty neck (as when the upper level
        // has a smaller footprint). Level 1: floor/ceiling slabs at z∈[0,.5] and
        // [2.5,3] with dense walls between; EMPTY neck z∈[3,4]; level 2:
        // [4,4.5] and [5.5,6] with walls. The neck is the only near-empty valley.
        let mut pts = Vec::new();
        let mut s: u64 = 11;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut slab =
            |pts: &mut Vec<[f32; 3]>, z0: f32, z1: f32, n: usize, rnd: &mut dyn FnMut() -> f32| {
                for _ in 0..n {
                    pts.push([rnd() * 4.0, rnd() * 4.0, z0 + (z1 - z0) * rnd()]);
                }
            };
        slab(&mut pts, 0.0, 0.5, 2000, &mut rnd); // floor 1
        slab(&mut pts, 2.5, 3.0, 2000, &mut rnd); // ceiling 1
        slab(&mut pts, 0.5, 2.5, 2600, &mut rnd); // walls 1 (dense interior)
        // empty neck z∈[3,4]
        slab(&mut pts, 4.0, 4.5, 2000, &mut rnd); // floor 2
        slab(&mut pts, 5.5, 6.0, 2000, &mut rnd); // ceiling 2
        slab(&mut pts, 4.5, 5.5, 1300, &mut rnd); // walls 2

        // min_height ≈ room scale: heavy-smooth merges each level's internal
        // floor/interior/ceiling structure, and nearby drops collapse to one.
        let storeys = find_storeys(&pts, [0.0, 0.0, 1.0], 0.2, 2.0);
        assert_eq!(storeys.len(), 2, "expected 2 storeys, got {storeys:?}");
        // Split falls at/near the neck z∈[3,4].
        assert!(
            (2.5..4.1).contains(&storeys[0].1),
            "split should be near the neck: {storeys:?}"
        );
    }

    #[test]
    fn manhattan_finds_orthogonal_planes() {
        // A box corner: floor (z=0), and two perpendicular walls (x=0, y=0).
        // Greedy peeling would over-pick the largest; Manhattan must return
        // planes in all THREE orientations.
        let mut pts = Vec::new();
        let mut s: u64 = 3;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32)
        };
        for _ in 0..1500 {
            pts.push([rnd(), rnd(), 0.01 * rnd()]);
        } // z=0 floor (big)
        for _ in 0..600 {
            pts.push([0.01 * rnd(), rnd(), rnd()]);
        } // x=0 wall
        for _ in 0..600 {
            pts.push([rnd(), 0.01 * rnd(), rnd()]);
        } // y=0 wall

        let est = ManhattanPlanes {
            k: 20,
            dist_thresh: 0.05,
            angle_thresh: 30f32.to_radians(),
            min_support: 100,
        };
        let planes = est.estimate(&pts);
        assert!(
            planes.len() >= 3,
            "want ≥3 planes across orientations, got {}",
            planes.len()
        );
        // Normals should span all three axes (up to sign), not one direction.
        let axis_of = |n: &[f32; 3]| {
            let a = [n[0].abs(), n[1].abs(), n[2].abs()];
            if a[0] >= a[1] && a[0] >= a[2] {
                0
            } else if a[1] >= a[2] {
                1
            } else {
                2
            }
        };
        let mut seen = [false; 3];
        for p in &planes {
            seen[axis_of(&p.0)] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "planes should cover all 3 axes, got {seen:?}"
        );
    }

    #[test]
    fn global_peeling_covers_disjoint_planes() {
        // Two coplanar patches separated by a gap (a "holey wall") plus a
        // perpendicular wall. Region growing would split the gapped plane;
        // global peeling must recover it as ONE plane and cover most points.
        let mut pts = Vec::new();
        let mut s: u64 = 7;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        // z=0 plane, two patches with a gap in x (disjoint).
        for _ in 0..400 {
            pts.push([rnd() * 0.4 - 0.6, rnd(), 0.01 * rnd()]);
        }
        for _ in 0..400 {
            pts.push([rnd() * 0.4 + 0.6, rnd(), 0.01 * rnd()]);
        }
        // x=1 perpendicular wall.
        for _ in 0..400 {
            pts.push([1.0 + 0.01 * rnd(), rnd(), rnd()]);
        }

        let est = GlobalPlanePeeling {
            k: 20,
            normal_consensus: false,
            dist_thresh: 0.05,
            angle_thresh: 20f32.to_radians(),
            min_support: 50,
            max_planes: 10,
            max_iterations: 500,
            confidence: 0.99,
        };
        let planes = est.estimate(&pts);
        assert!(planes.len() >= 2, "want ≥2 planes, got {}", planes.len());
        // The gapped z=0 plane should be a single segment spanning both patches.
        let biggest = &planes[0];
        assert!(
            biggest.2.len() >= 700,
            "largest plane should unify both disjoint patches (~800), got {}",
            biggest.2.len()
        );
        let covered: usize = planes.iter().map(|p| p.2.len()).sum();
        assert!(
            covered as f32 / pts.len() as f32 > 0.8,
            "coverage {covered}/{}",
            pts.len()
        );
    }

    #[test]
    fn footprint_separates_exterior_and_interior_walls() {
        // A solid (filled) unit-square footprint = one room full of points.
        let mut fp_pts = Vec::new();
        let mut y = 0.0;
        while y <= 1.0 {
            let mut x = 0.0;
            while x <= 1.0 {
                fp_pts.push((x, y));
                x += 0.05;
            }
            y += 0.05;
        }
        let fp = build_footprint2d(&fp_pts, 0.1, 0);
        // Left edge wall: exterior empty on one side, room on the other.
        let left: Vec<(f32, f32)> = (0..20).map(|i| (0.0, i as f32 * 0.05)).collect();
        assert!(
            fp.wall_is_exterior(&left, (1.0, 0.0)),
            "outer edge must read exterior"
        );
        // Interior partition down the middle: room on both sides.
        let mid: Vec<(f32, f32)> = (0..20).map(|i| (0.5, i as f32 * 0.05)).collect();
        assert!(
            !fp.wall_is_exterior(&mid, (1.0, 0.0)),
            "interior partition must read interior"
        );
    }
}
