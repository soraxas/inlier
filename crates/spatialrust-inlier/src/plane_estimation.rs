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
//!
//! Future methods (e.g. global normal-consensus plane peeling for disjoint
//! points) implement the same trait and slot in without touching callers.

use crate::region_growing::{region_growing_ransac_with_progress, RansacMode};

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
}
