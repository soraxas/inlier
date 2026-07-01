//! High-level pipeline API for the dollhouse cutaway effect.
//!
//! This module provides a single entry point, [`segment_for_dollhouse`], that
//! runs the full Ling et al. 2024 plane-segmentation pipeline and returns a
//! [`DollhouseScene`] ready to drive the dollhouse rendering step.
//!
//! ## What the pipeline does
//!
//! ```text
//! pts  ──► region_growing_ransac    (normals → region grow → RANSAC per cluster)
//!       ──► merge_planes            (union-find collapse of co-planar fragments)
//!       ──► grow_planes ×N          (absorb unassigned leftovers, refit)
//!       ──► canonicalize normals    (orient each normal inward toward cloud bulk)
//!       ──► classify exterior       (outward fraction below threshold → exterior)
//!       ──► DollhouseScene          (planes + unassigned indices, ready to render)
//! ```
//!
//! Each [`SegmentedPlane`] carries a **canonical inward normal** — oriented
//! toward the majority of the cloud — which makes the per-frame dollhouse dot
//! product unambiguous:
//!
//! ```text
//! dot = canonical_inward_normal · normalize(cam_pos − centroid)
//! hide if dot < −cos(dollhouse_angle)
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! use spatialrust_inlier::dollhouse::{DollhouseParams, segment_for_dollhouse};
//!
//! # let pts: Vec<[f32; 3]> = vec![];
//! let params = DollhouseParams::default();
//! let scene = segment_for_dollhouse(&pts, &params);
//! println!("{} planes, {} unassigned", scene.planes.len(), scene.unassigned_indices.len());
//! ```

use crate::region_growing::{region_growing_ransac, RansacMode};
use crate::plane_ops::{merge_planes, grow_planes, GrowArgs};

/// Full parameter set for the dollhouse segmentation pipeline.
///
/// All defaults are tuned for indoor building scans at roughly 1–5 cm point
/// spacing and 2–8 mm measurement noise (typical of structured-light or
/// medium-grade LiDAR).  Use [`auto_tune_settings`](crate::auto_tune::auto_tune_settings)
/// to derive per-cloud estimates.
#[derive(Debug, Clone)]
pub struct DollhouseParams {
    // ── Segmentation (region growing + RANSAC) ──────────────────────────────
    /// k for normal/curvature k-NN (default 20).
    pub k_neighbors: usize,
    /// Max normal deviation in region growing, **radians** (default 10°).
    pub angle_thresh: f32,
    /// Minimum region-growing cluster size (default 30).
    pub min_cluster_size: usize,
    /// RANSAC inlier distance band (default 0.08 m).
    pub dist_thresh: f32,
    /// Which RANSAC scorer to use (default Simple).
    pub ransac_mode: RansacMode,
    /// MAGSAC++ σ_max (default `dist_thresh × 1.5`; ignored for Simple/MSAC).
    pub sigma_max: f64,
    /// MSAC / MAGSAC++ iteration cap (default 1000).
    pub max_iterations: usize,
    /// MSAC / MAGSAC++ confidence (default 0.99).
    pub confidence: f64,

    // ── Merge ────────────────────────────────────────────────────────────────
    /// Max normal angle for merge co-planarity, **radians** (default 5°).
    pub merge_angle_thresh: f32,
    /// Max plane-offset difference for merge (default 0.15 m).
    pub merge_dist_thresh: f32,
    /// Drop merged plane if fewer inliers (default 200).
    pub merge_min_pts: usize,

    // ── Grow ─────────────────────────────────────────────────────────────────
    /// Max point-to-plane distance in grow step (default 0.20 m).
    pub grow_dist_thresh: f32,
    /// Max grow-refit iterations (default 3).
    pub grow_max_iters: usize,
    /// Enable normal agreement filter in grow (default true).
    pub grow_use_normal: bool,
    /// Cosine of max allowed angle between point normal and plane normal in grow
    /// (default `cos(30°) ≈ 0.866`).
    pub grow_normal_cos_thresh: f32,
    /// Enable curvature filter in grow (default true).
    pub grow_use_curvature: bool,
    /// Max local curvature to allow absorption (default 0.05).
    pub grow_max_curvature: f32,
    /// Enable connectivity filter in grow (default true).
    pub grow_use_connectivity: bool,

    // ── Normal canonicalization / exterior classification ────────────────────
    /// Margin around the plane (multiples of `dist_thresh`) within which points
    /// are excluded from the inward/outward vote — prevents near-plane noise
    /// from inflating the outward count (default 3.0).
    pub canonicalize_margin_factor: f32,
    /// Fraction of far-side points on the outward side below which a plane is
    /// classified as an exterior surface (default 0.05).
    pub exterior_thresh: f32,
}

impl Default for DollhouseParams {
    fn default() -> Self {
        Self {
            k_neighbors: 20,
            angle_thresh: 10f32.to_radians(),
            min_cluster_size: 30,
            dist_thresh: 0.08,
            ransac_mode: RansacMode::Simple,
            sigma_max: 0.0,
            max_iterations: 1000,
            confidence: 0.99,
            merge_angle_thresh: 5f32.to_radians(),
            merge_dist_thresh: 0.15,
            merge_min_pts: 200,
            grow_dist_thresh: 0.20,
            grow_max_iters: 3,
            grow_use_normal: true,
            grow_normal_cos_thresh: 30f32.to_radians().cos(),
            grow_use_curvature: true,
            grow_max_curvature: 0.05,
            grow_use_connectivity: true,
            canonicalize_margin_factor: 3.0,
            exterior_thresh: 0.05,
        }
    }
}

/// A single segmented plane, ready for dollhouse rendering.
#[derive(Debug, Clone)]
pub struct SegmentedPlane {
    /// Unit normal oriented **inward** (toward the majority of cloud points).
    /// The dollhouse dot-product test is unambiguous with this orientation:
    /// `hide if normal · normalize(cam_pos − centroid) < −cos(dollhouse_angle)`.
    pub normal: [f32; 3],
    /// Plane offset such that `normal · p + d ≈ 0` for inlier points.
    pub d: f32,
    /// Indices into the `pts` slice passed to [`segment_for_dollhouse`].
    pub inlier_indices: Vec<usize>,
    /// Centroid of the inlier point set.
    pub centroid: [f32; 3],
    /// `true` if fewer than `exterior_thresh` fraction of cloud points lie
    /// unambiguously outside this plane (i.e. on the air side).  Exterior
    /// planes are outer walls / roof / floor; interior planes are partitions.
    pub is_exterior: bool,
}

/// Output of the full dollhouse segmentation pipeline.
#[derive(Debug, Clone)]
pub struct DollhouseScene {
    /// Segmented planes, sorted by inlier count descending.
    pub planes: Vec<SegmentedPlane>,
    /// Indices of points not assigned to any plane (furniture, clutter, noise).
    pub unassigned_indices: Vec<usize>,
}

/// Run the full segmentation pipeline and return a [`DollhouseScene`].
///
/// This is the main entry point for the library.  It chains:
/// 1. [`region_growing_ransac`] — normals + region grow + RANSAC per cluster
/// 2. [`merge_planes`] — union-find collapse of co-planar fragments
/// 3. [`grow_planes`] × `params.grow_max_iters` — absorb unassigned leftovers
/// 4. Normal canonicalization (inward orientation via majority-side vote)
/// 5. Exterior classification (`outward_frac < exterior_thresh`)
///
/// # Panics
/// Does not panic.  Returns an empty scene for degenerate input (< 3 points).
pub fn segment_for_dollhouse(pts: &[[f32; 3]], params: &DollhouseParams) -> DollhouseScene {
    if pts.len() < 3 {
        return DollhouseScene { planes: vec![], unassigned_indices: (0..pts.len()).collect() };
    }

    let sigma_max = if params.sigma_max > 0.0 {
        params.sigma_max
    } else {
        (params.dist_thresh * 1.5) as f64
    };

    // Step 1: segment.
    let raw_planes = region_growing_ransac(
        pts,
        params.k_neighbors,
        params.angle_thresh,
        params.min_cluster_size,
        params.dist_thresh,
        params.ransac_mode,
        sigma_max,
        params.max_iterations,
        params.confidence,
    );

    // Step 2: merge.
    let mut merged = merge_planes(
        &raw_planes,
        pts,
        params.merge_angle_thresh,
        params.merge_dist_thresh,
        params.merge_min_pts,
    );

    // Step 3: grow (iterate until stable or max_iters).
    if params.grow_max_iters > 0 && !merged.is_empty() {
        let grow_args = GrowArgs {
            dist_thresh: params.grow_dist_thresh,
            use_normal: params.grow_use_normal,
            normal_cos_thresh: params.grow_normal_cos_thresh,
            use_curvature: params.grow_use_curvature,
            max_curvature: params.grow_max_curvature,
            use_connectivity: params.grow_use_connectivity,
        };
        let mut grown = grow_planes(&merged, pts, &grow_args);
        for _ in 1..params.grow_max_iters {
            let prev: usize = grown.iter().map(|(_, _, v)| v.len()).sum();
            grown = grow_planes(&grown, pts, &grow_args);
            let next: usize = grown.iter().map(|(_, _, v)| v.len()).sum();
            if next == prev {
                break;
            }
        }
        merged = grown;
    }

    // Steps 4 + 5: canonicalize normals and classify exterior.
    let margin = params.dist_thresh * params.canonicalize_margin_factor;
    let mut result_planes: Vec<SegmentedPlane> = Vec::with_capacity(merged.len());

    for (normal, d, inliers) in &merged {
        let (canon_normal, canon_d, centroid, is_exterior) =
            classify_plane(*normal, *d, inliers, pts, margin, params.exterior_thresh);
        result_planes.push(SegmentedPlane {
            normal: canon_normal,
            d: canon_d,
            inlier_indices: inliers.clone(),
            centroid,
            is_exterior,
        });
    }

    // Unassigned indices.
    let mut inlier_flags = vec![false; pts.len()];
    for plane in &result_planes {
        for &i in &plane.inlier_indices {
            if i < inlier_flags.len() {
                inlier_flags[i] = true;
            }
        }
    }
    let unassigned_indices: Vec<usize> = (0..pts.len())
        .filter(|&i| !inlier_flags[i])
        .collect();

    DollhouseScene {
        planes: result_planes,
        unassigned_indices,
    }
}

/// Canonicalize a plane's normal to point **inward** (toward the bulk of the
/// cloud) and classify whether it is an exterior surface.
///
/// This is the shared implementation of steps 4 + 5 of the pipeline, factored
/// out so interactive consumers (which run segment / merge / grow as separate
/// user-triggered steps) classify planes through the exact same code path as
/// [`segment_for_dollhouse`].
///
/// - `margin`: half-width of the near-plane band excluded from the inward/outward
///   vote (typically `dist_thresh × canonicalize_margin_factor`).
/// - `exterior_thresh`: outward-fraction below which the plane is exterior.
///
/// Returns `(canonical_normal, canonical_d, centroid, is_exterior)`. The
/// returned normal/`d` satisfy `canonical_normal · p + canonical_d ≈ 0` on the
/// plane, with the normal oriented toward the interior side.
pub fn classify_plane(
    normal: [f32; 3],
    d: f32,
    inliers: &[usize],
    pts: &[[f32; 3]],
    margin: f32,
    exterior_thresh: f32,
) -> ([f32; 3], f32, [f32; 3], bool) {
    // Centroid of inlier set.
    let centroid = if inliers.is_empty() {
        [0.0f32; 3]
    } else {
        let n = inliers.len() as f32;
        let (mut cx, mut cy, mut cz) = (0f32, 0f32, 0f32);
        for &i in inliers {
            cx += pts[i][0];
            cy += pts[i][1];
            cz += pts[i][2];
        }
        [cx / n, cy / n, cz / n]
    };

    // Count points clearly on each side of the plane (exclude near-plane band).
    let mut far_pos = 0usize;
    let mut far_neg = 0usize;
    for &p in pts {
        let signed = normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + d;
        if signed > margin {
            far_pos += 1;
        } else if signed < -margin {
            far_neg += 1;
        }
    }

    // Canonical normal points toward the larger (interior) side.
    let (canon_normal, canon_d) = if far_pos >= far_neg {
        (normal, d)
    } else {
        ([-normal[0], -normal[1], -normal[2]], -d)
    };

    // Exterior: the minority (outward/air) side is very small.
    let total_far = (far_pos + far_neg).max(1);
    let outward_far = far_pos.min(far_neg);
    let outward_frac = outward_far as f32 / total_far as f32;
    let is_exterior = outward_frac < exterior_thresh;

    (canon_normal, canon_d, centroid, is_exterior)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normals::{normalize3, cross3};

    fn synthetic_planes(seed: u32) -> Vec<[f32; 3]> {
        let mut s = seed as u64;
        let mut rng = move || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let mut out = Vec::new();
        for pi in 0..3 {
            let normal = loop {
                let (x, y, z) = (rng(), rng(), rng());
                let len = (x*x+y*y+z*z).sqrt();
                if len > 0.01 && len <= 1.0 { break normalize3([x, y, z]); }
            };
            let offset = pi as f32 * 2.5;
            let p0 = [-offset*normal[0], -offset*normal[1], -offset*normal[2]];
            let up = if normal[2].abs() < 0.9 { [0f32,0.,1.] } else { [1.,0.,0.] };
            let t1 = normalize3(cross3(normal, up));
            let t2 = cross3(normal, t1);
            for _ in 0..600 {
                let u = rng() * 3.0;
                let v = rng() * 3.0;
                let noise = rng() * 0.03;
                out.push([
                    p0[0] + u*t1[0] + v*t2[0] + noise*normal[0],
                    p0[1] + u*t1[1] + v*t2[1] + noise*normal[1],
                    p0[2] + u*t1[2] + v*t2[2] + noise*normal[2],
                ]);
            }
        }
        out
    }

    #[test]
    fn pipeline_finds_three_planes() {
        let pts = synthetic_planes(42);
        let scene = segment_for_dollhouse(&pts, &DollhouseParams::default());
        assert!(scene.planes.len() >= 2,
            "expected ≥2 planes, got {}", scene.planes.len());
        // All inlier normals should be unit-length.
        for p in &scene.planes {
            let len = (p.normal[0].powi(2) + p.normal[1].powi(2) + p.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "normal not unit: {len}");
        }
        // Total inliers + unassigned = total points.
        let total_inliers: usize = scene.planes.iter().map(|p| p.inlier_indices.len()).sum();
        assert_eq!(total_inliers + scene.unassigned_indices.len(), pts.len());
    }

    #[test]
    fn empty_input_ok() {
        let scene = segment_for_dollhouse(&[], &DollhouseParams::default());
        assert!(scene.planes.is_empty());
        assert!(scene.unassigned_indices.is_empty());
    }
}
