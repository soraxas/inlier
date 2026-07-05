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

use crate::plane::fit_plane_msac;
use crate::region_growing::{region_growing_ransac_with_progress, RansacMode};
use crate::spatial_grid::{build_grid, estimate_cell_size, knn};
use crate::normals::pca_normal_and_curvature;

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
                        pca_normal_and_curvature(pts, &nb).map(|(nv, _)| nv).unwrap_or([0.0; 3])
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
            on_progress(0.2 + 0.8 * (1.0 - remaining.len() as f32 / n as f32), "Peeling planes");
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
    let a = if up[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
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
fn cluster_1d_peaks(mut items: Vec<(f32, usize)>, dist: f32, min_support: usize) -> Vec<(f32, Vec<usize>)> {
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
        if c >= min_support
            && (i == 0 || c >= counts[i - 1])
            && (i + 1 == nb || c >= counts[i + 1])
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
        let cell = estimate_cell_size(pts);
        let grid = build_grid(pts, cell);
        let normals: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let nb = knn(pts, i, self.k, cell, &grid);
                if nb.len() < 3 {
                    [0.0; 3]
                } else {
                    pca_normal_and_curvature(pts, &nb).map(|(nv, _)| nv).unwrap_or([0.0; 3])
                }
            })
            .collect();

        // 2. Orthogonal frame from normal statistics.
        on_progress(0.5, "Estimating frame");
        let up = dominant_direction(&normals, None);
        let h1 = dominant_horizontal(&normals, up);
        let mut h2 = cross3(up, h1);
        let h2n = (h2[0] * h2[0] + h2[1] * h2[1] + h2[2] * h2[2]).sqrt();
        if h2n > 1e-6 {
            h2 = [h2[0] / h2n, h2[1] / h2n, h2[2] / h2n];
        }
        let axes = [up, h1, h2];

        // 3. Orientation vote: assign each point to the axis its normal matches.
        on_progress(0.6, "Extracting planes");
        let cos_t = self.angle_thresh.cos();
        let mut groups: [Vec<(f32, usize)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..n {
            let nv = normals[i];
            if nv == [0.0; 3] {
                continue;
            }
            let d = [dot3(nv, axes[0]).abs(), dot3(nv, axes[1]).abs(), dot3(nv, axes[2]).abs()];
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
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        for _ in 0..600 { pts.push([rnd(), rnd(), 0.0 + 0.01 * rnd()]); }
        for _ in 0..600 { pts.push([rnd(), 0.0 + 0.01 * rnd(), rnd()]); }
        for _ in 0..600 { pts.push([0.0 + 0.01 * rnd(), rnd(), rnd()]); }

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
            &pts, est.k, est.angle_thresh, est.min_cluster_size, est.dist_thresh,
            est.mode, est.sigma_max, est.max_iterations, est.confidence,
        );
        assert_eq!(via_trait.len(), via_free.len());
        for (a, b) in via_trait.iter().zip(via_free.iter()) {
            assert_eq!(a.2, b.2);
        }
    }

    #[test]
    fn manhattan_finds_orthogonal_planes() {
        // A box corner: floor (z=0), and two perpendicular walls (x=0, y=0).
        // Greedy peeling would over-pick the largest; Manhattan must return
        // planes in all THREE orientations.
        let mut pts = Vec::new();
        let mut s: u64 = 3;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32)
        };
        for _ in 0..1500 { pts.push([rnd(), rnd(), 0.01 * rnd()]); } // z=0 floor (big)
        for _ in 0..600 { pts.push([0.01 * rnd(), rnd(), rnd()]); } // x=0 wall
        for _ in 0..600 { pts.push([rnd(), 0.01 * rnd(), rnd()]); } // y=0 wall

        let est = ManhattanPlanes {
            k: 20,
            dist_thresh: 0.05,
            angle_thresh: 30f32.to_radians(),
            min_support: 100,
        };
        let planes = est.estimate(&pts);
        assert!(planes.len() >= 3, "want ≥3 planes across orientations, got {}", planes.len());
        // Normals should span all three axes (up to sign), not one direction.
        let axis_of = |n: &[f32; 3]| {
            let a = [n[0].abs(), n[1].abs(), n[2].abs()];
            if a[0] >= a[1] && a[0] >= a[2] { 0 } else if a[1] >= a[2] { 1 } else { 2 }
        };
        let mut seen = [false; 3];
        for p in &planes {
            seen[axis_of(&p.0)] = true;
        }
        assert!(seen.iter().all(|&s| s), "planes should cover all 3 axes, got {seen:?}");
    }

    #[test]
    fn global_peeling_covers_disjoint_planes() {
        // Two coplanar patches separated by a gap (a "holey wall") plus a
        // perpendicular wall. Region growing would split the gapped plane;
        // global peeling must recover it as ONE plane and cover most points.
        let mut pts = Vec::new();
        let mut s: u64 = 7;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        // z=0 plane, two patches with a gap in x (disjoint).
        for _ in 0..400 { pts.push([rnd() * 0.4 - 0.6, rnd(), 0.01 * rnd()]); }
        for _ in 0..400 { pts.push([rnd() * 0.4 + 0.6, rnd(), 0.01 * rnd()]); }
        // x=1 perpendicular wall.
        for _ in 0..400 { pts.push([1.0 + 0.01 * rnd(), rnd(), rnd()]); }

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
        assert!(covered as f32 / pts.len() as f32 > 0.8, "coverage {covered}/{}", pts.len());
    }
}
