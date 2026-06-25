//! PCA-based surface normal and curvature estimation for unstructured point clouds.
//!
//! ## Algorithm overview
//!
//! For each point `i` and its k nearest neighbours, we:
//! 1. Compute the 3×3 covariance matrix of the neighbourhood.
//! 2. Find the **smallest** eigenvector — the direction of least variance — which
//!    is the surface normal.
//! 3. Compute local **curvature** as `λ_min / trace(C)` (Pauly et al. 2002).
//!
//! ## Numerical method: shift trick + power iteration
//!
//! Direct smallest-eigenvector extraction is numerically tricky.  Instead we use
//! the **shift trick**: form `B = trace(C)·I − C`.  The *largest* eigenvector of
//! `B` equals the *smallest* eigenvector of `C`.  Power iteration (20 steps) on
//! the 3×3 matrix `B` converges reliably.  Curvature is recovered via the Rayleigh
//! quotient: `λ_min = v^T C v` (v already unit length after power iteration).
//!
//! ## PCA plane helpers
//!
//! [`fit_plane_3pts`] and [`fit_plane_ls`] (least-squares via PCA) are also here
//! because they share the same covariance/normal machinery.

/// Extract surface normal and local curvature from a k-NN neighbourhood via PCA.
///
/// `pts` is the full point cloud; `idxs` contains the indices of the neighbourhood
/// (including the query point itself — the centroid subtraction handles it).
///
/// Returns `(unit_normal, curvature)`:
/// - `unit_normal` — smallest eigenvector of the covariance matrix (surface tangent
///   plane normal); sign is arbitrary (see [`spatialrust_inlier::dollhouse`] for
///   canonicalization).
/// - `curvature` — dimensionless ratio `λ_min / trace ∈ [0, 1]`.
///   Flat surfaces → ≈ 0; edges, furniture curves → > 0.05.
///
/// Returns `None` if fewer than 3 points or if the neighbourhood is degenerate
/// (all points coincident).
pub fn pca_normal_and_curvature(pts: &[[f32; 3]], idxs: &[usize]) -> Option<([f32; 3], f32)> {
    let n = idxs.len();
    if n < 3 {
        return None;
    }
    let nf = n as f32;

    // Centroid.
    let mut cx = 0f32;
    let mut cy = 0f32;
    let mut cz = 0f32;
    for &i in idxs {
        cx += pts[i][0];
        cy += pts[i][1];
        cz += pts[i][2];
    }
    cx /= nf;
    cy /= nf;
    cz /= nf;

    // Upper-triangle covariance (symmetric).
    let mut cov = [[0f32; 3]; 3];
    for &i in idxs {
        let dx = pts[i][0] - cx;
        let dy = pts[i][1] - cy;
        let dz = pts[i][2] - cz;
        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[0][2] += dx * dz;
        cov[1][1] += dy * dy;
        cov[1][2] += dy * dz;
        cov[2][2] += dz * dz;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    if trace < 1e-12 {
        return None;
    }

    // Shift trick: B = trace·I − C.  Largest eigenvector of B = smallest of C.
    let b = [
        [trace - cov[0][0], -cov[0][1], -cov[0][2]],
        [-cov[1][0], trace - cov[1][1], -cov[1][2]],
        [-cov[2][0], -cov[2][1], trace - cov[2][2]],
    ];
    let v = power_iter(b, [0.1, 0.5, 0.9]);
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-6 {
        return None;
    }
    let normal = [v[0] / len, v[1] / len, v[2] / len];

    // Curvature = λ_min / trace  via Rayleigh quotient.
    let cv = matvec(cov, normal);
    let lambda_min = normal[0] * cv[0] + normal[1] * cv[1] + normal[2] * cv[2];
    let curvature = (lambda_min / trace).max(0.0);

    Some((normal, curvature))
}

/// Power iteration to find the dominant eigenvector of a 3×3 symmetric matrix.
///
/// Runs for 20 iterations with seed `init`.  Convergence for 3×3 matrices is
/// typically achieved in 8–12 iterations; 20 provides a safety margin.
pub fn power_iter(m: [[f32; 3]; 3], init: [f32; 3]) -> [f32; 3] {
    let mut v = normalize3(init);
    for _ in 0..20 {
        let w = matvec(m, v);
        let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if len < 1e-10 {
            break;
        }
        v = [w[0] / len, w[1] / len, w[2] / len];
    }
    v
}

/// Multiply a 3×3 matrix by a 3-vector.
#[inline]
pub fn matvec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Normalize a 3-vector; returns `[1, 0, 0]` if the input length is below `1e-8`.
#[inline]
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Cross product of two 3-vectors.
#[inline]
pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Fit a plane through exactly three non-collinear points.
///
/// Returns `(unit_normal, d)` such that `normal · p + d = 0`, or `None` if the
/// points are collinear (cross product length < `1e-8`).
pub fn fit_plane_3pts(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<([f32; 3], f32)> {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = cross3(ab, ac);
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-8 {
        return None;
    }
    let n = [n[0] / len, n[1] / len, n[2] / len];
    let d = -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2]);
    Some((n, d))
}

/// Least-squares plane fit via PCA on an inlier set.
///
/// The normal is the smallest PCA eigenvector of the inlier covariance, and
/// `d` is computed from the centroid so that `normal · centroid + d = 0`.
///
/// Returns `None` if fewer than 3 points or if the neighbourhood is degenerate.
pub fn fit_plane_ls(pts: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
    if pts.len() < 3 {
        return None;
    }
    let idxs: Vec<usize> = (0..pts.len()).collect();
    let (normal, _) = pca_normal_and_curvature(pts, &idxs)?;
    let n = pts.len() as f32;
    let cx = pts.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f32>() / n;
    let cz = pts.iter().map(|p| p[2]).sum::<f32>() / n;
    let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
    Some((normal, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_flat_plane() {
        // XY-plane cluster → normal should be ≈ [0, 0, 1].
        let pts: Vec<[f32; 3]> = (0..25).map(|i| {
            let x = (i % 5) as f32 * 0.1;
            let y = (i / 5) as f32 * 0.1;
            [x, y, 0.0]
        }).collect();
        let idxs: Vec<usize> = (0..pts.len()).collect();
        let (n, curv) = pca_normal_and_curvature(&pts, &idxs).unwrap();
        assert!(n[2].abs() > 0.99, "normal z should ≈ ±1, got {n:?}");
        assert!(curv < 0.01, "curvature should be ≈ 0, got {curv}");
    }

    #[test]
    fn fit_3pts_known() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let (n, d) = fit_plane_3pts(a, b, c).unwrap();
        assert!((n[2].abs() - 1.0).abs() < 1e-5, "z-plane normal: {n:?}");
        assert!(d.abs() < 1e-5, "d should be 0: {d}");
    }
}
