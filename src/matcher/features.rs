//! FasterPFH (Fast Point Feature Histogram) feature extraction
//!
//! Implements the FasterPFH descriptor from KISS-Matcher:
//! 1. Normal estimation using PCA
//! 2. Geometric suppression (linearity filtering)
//! 3. SPFH (Simplified Point Feature Histogram)
//! 4. FPFH (Fast Point Feature Histogram) aggregation
//!
//! Reference: KISS-Matcher paper and PCL FPFH implementation
//! FasterPFH feature extraction for point clouds
//!
//! This module implements the FasterPFH (Fast Point Feature Histogram) descriptor
//! from the KISS-Matcher paper. It extracts geometric features from 3D point clouds
//! for robust correspondence matching.
//!
//! # Algorithm Overview
//!
//! 1. **Normal Estimation**: Use PCA on local neighborhoods to estimate surface normals
//! 2. **Linearity Filtering**: Remove edge-like features (keep planar regions)
//! 3. **SPFH Computation**: Compute Simplified Point Feature Histograms
//! 4. **FPFH Aggregation**: Aggregate SPFH from neighbors into final descriptor
//!
//! # Example
//!
//! ```ignore
//! use inlier::matcher::features::FasterPFH;
//! use inlier::types::DataMatrix;
//!
//! // Create point cloud (N points × 3 dimensions)
//! let points = DataMatrix::from_row_slice(100, 3, &data);
//!
//! // Extract features
//! let fpfh = FasterPFH::new(
//!     0.2,   // normal_radius: radius for normal estimation
//!     0.35,  // fpfh_radius: radius for FPFH computation
//!     0.9,   // the_linearity: filter threshold (0-1)
//!     11,    // bins: histogram bins per feature
//! );
//! let features = fpfh.compute_features(&points);
//!
//! // Use features for matching...
//! ```

use crate::types::DataMatrix;
use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// FasterPFH descriptor extractor
pub struct FasterPFH {
    /// Radius for normal estimation
    pub normal_radius: f64,
    /// Radius for FPFH feature computation
    pub fpfh_radius: f64,
    /// Linearity threshold for filtering planar regions (1.0 = disabled)
    pub the_linearity: f64,
    /// Number of bins per angular feature (typically 11)
    pub bins: usize,
}

/// Point with normal and descriptor
#[derive(Clone, Debug)]
pub struct FeaturePoint {
    pub point: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub descriptor: Vec<f64>, // FPFH descriptor (33 values = 3 * 11 bins)
    pub is_valid: bool,
}

impl FasterPFH {
    /// Create new FasterPFH extractor
    pub fn new(normal_radius: f64, fpfh_radius: f64, the_linearity: f64, bins: usize) -> Self {
        Self {
            normal_radius,
            fpfh_radius,
            the_linearity,
            bins,
        }
    }

    /// Extract features from point cloud
    ///
    /// # Returns
    /// Vector of feature points (only valid points with good normals)
    pub fn compute_features(&self, points: &DataMatrix) -> Vec<FeaturePoint> {
        let n = points.n_points();
        if n < 3 {
            return Vec::new();
        }

        // Build KD-tree for neighbor search
        let kdtree = self.build_kdtree(points);

        // Step 1: Estimate normals with linearity filtering
        #[cfg(feature = "progress")]
        let pb = crate::progress::create_progress_bar(n as u64, "Computing normals");

        #[cfg(feature = "rayon")]
        let iter = (0..n).into_par_iter();
        #[cfg(not(feature = "rayon"))]
        let iter = 0..n;

        let features: Vec<_> = iter
            .filter_map(|i| {
                #[cfg(feature = "progress")]
                pb.inc(1);

                let point = Vector3::new(points.get(i, 0), points.get(i, 1), points.get(i, 2));

                // Find neighbors for normal estimation
                let neighbors = self.radius_search(&kdtree, &point, self.normal_radius);
                if neighbors.len() < 3 {
                    return None; // Need at least 3 points for PCA
                }

                // Estimate normal using PCA
                let (normal, linearity, _eigenvalues) =
                    self.estimate_normal_with_linearity(points, &neighbors);

                // Filter by linearity (keep points where linearity is LOW, i.e., more planar)
                // High linearity means linear/edge-like features
                if linearity > self.the_linearity {
                    return None; // Too linear (edges), skip
                }

                // Check normal validity
                if !Self::is_normal_valid(&normal) {
                    return None;
                }

                Some(FeaturePoint {
                    point,
                    normal,
                    descriptor: vec![0.0; 3 * self.bins], // Placeholder
                    is_valid: true,
                })
            })
            .collect();

        #[cfg(feature = "progress")]
        pb.finish();

        if features.is_empty() {
            return features;
        }

        // Step 2: Compute FPFH descriptors
        let mut features = features; // Make mutable for descriptor computation
        self.compute_fpfh_descriptors(&mut features, &kdtree, points);

        features
    }

    /// Build KD-tree for fast neighbor queries
    fn build_kdtree(&self, points: &DataMatrix) -> KdTree<f64, 3> {
        let items: Vec<[f64; 3]> = (0..points.n_points())
            .map(|i| [points.get(i, 0), points.get(i, 1), points.get(i, 2)])
            .collect();
        (&items).into()
    }

    /// Radius search for neighbors
    fn radius_search(
        &self,
        kdtree: &KdTree<f64, 3>,
        point: &Vector3<f64>,
        radius: f64,
    ) -> Vec<usize> {
        let query = [point.x, point.y, point.z];

        kdtree
            .within::<SquaredEuclidean>(&query, radius * radius)
            .iter()
            .map(|nn| nn.item as usize)
            .collect()
    }

    /// Estimate normal using PCA and return linearity measure
    ///
    /// Linearity = (λ1 - λ2) / λ1 where λ1 >= λ2 >= λ3 are eigenvalues
    fn estimate_normal_with_linearity(
        &self,
        points: &DataMatrix,
        neighbors: &[usize],
    ) -> (Vector3<f64>, f64, Vector3<f64>) {
        if neighbors.len() < 3 {
            return (Vector3::zeros(), 1.0, Vector3::zeros());
        }

        // Compute centroid
        let mut centroid = Vector3::zeros();
        for &idx in neighbors {
            centroid.x += points.get(idx, 0);
            centroid.y += points.get(idx, 1);
            centroid.z += points.get(idx, 2);
        }
        centroid /= neighbors.len() as f64;

        // Build covariance matrix
        let mut cov = Matrix3::zeros();
        for &idx in neighbors {
            let p = Vector3::new(
                points.get(idx, 0) - centroid.x,
                points.get(idx, 1) - centroid.y,
                points.get(idx, 2) - centroid.z,
            );
            cov += p * p.transpose();
        }
        cov /= neighbors.len() as f64;

        // Compute eigenvalues and eigenvectors
        let eigen = cov.symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues in descending order and track indices
        let mut indices = [0, 1, 2];
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

        // Eigenvalues in DESCENDING order: λ0 ≥ λ1 ≥ λ2
        // For a plane: λ0 ≈ λ1 >> λ2 (two large, one small)
        // For a line: λ0 >> λ1 ≈ λ2 (one large, two small)

        let idx_min = indices[2]; // Smallest eigenvalue - perpendicular to plane
        let normal = eigenvectors.column(idx_min).into_owned();

        // Linearity = (λ0 - λ1) / λ0
        // For plane: low value (λ0 ≈ λ1)
        // For line/edge: high value (λ0 >> λ1)
        let lambda0 = eigenvalues[indices[0]];
        let lambda1 = eigenvalues[indices[1]];
        let linearity = if lambda0 > 1e-10 {
            (lambda0 - lambda1) / lambda0
        } else {
            1.0
        };

        (
            normal,
            linearity,
            Vector3::new(lambda0, lambda1, eigenvalues[indices[2]]),
        )
    }

    /// Check if normal is valid (not NaN/inf)
    fn is_normal_valid(normal: &Vector3<f64>) -> bool {
        normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite() && normal.norm() > 1e-6
    }

    /// Compute FPFH descriptors for all features
    fn compute_fpfh_descriptors(
        &self,
        features: &mut [FeaturePoint],
        kdtree: &KdTree<f64, 3>,
        points: &DataMatrix,
    ) {
        // Build lookup from original point index to feature index
        let mut point_to_feature: HashMap<(i64, i64, i64), usize> = HashMap::new();
        for (feat_idx, feat) in features.iter().enumerate() {
            let key = (
                (feat.point.x * 1e6) as i64,
                (feat.point.y * 1e6) as i64,
                (feat.point.z * 1e6) as i64,
            );
            point_to_feature.insert(key, feat_idx);
        }

        // Compute SPFH for each feature
        #[cfg(feature = "progress")]
        let pb = crate::progress::create_progress_bar(features.len() as u64, "Computing SPFH");

        #[cfg(feature = "rayon")]
        use std::sync::Mutex;
        #[cfg(feature = "rayon")]
        let spfh_histograms: Vec<Mutex<[Vec<f64>; 3]>> = (0..features.len())
            .map(|_| {
                Mutex::new([
                    vec![0.0; self.bins],
                    vec![0.0; self.bins],
                    vec![0.0; self.bins],
                ])
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        let mut spfh_histograms: Vec<[Vec<f64>; 3]> = vec![
            [
                vec![0.0; self.bins],
                vec![0.0; self.bins],
                vec![0.0; self.bins]
            ];
            features.len()
        ];

        #[cfg(feature = "rayon")]
        {
            features.par_iter().enumerate().for_each(|(i, feat)| {
                #[cfg(feature = "progress")]
                pb.inc(1);

                let neighbors = self.radius_search(kdtree, &feat.point, self.fpfh_radius);

                for &n_idx in &neighbors {
                    let n_point = Vector3::new(
                        points.get(n_idx, 0),
                        points.get(n_idx, 1),
                        points.get(n_idx, 2),
                    );
                    let key = (
                        (n_point.x * 1e6) as i64,
                        (n_point.y * 1e6) as i64,
                        (n_point.z * 1e6) as i64,
                    );

                    if let Some(&n_feat_idx) = point_to_feature.get(&key) {
                        if i != n_feat_idx {
                            if let Some((f1, f2, f3)) = self.compute_pair_features(
                                &feat.point,
                                &feat.normal,
                                &features[n_feat_idx].point,
                                &features[n_feat_idx].normal,
                            ) {
                                let mut hist = spfh_histograms[i].lock().unwrap();
                                self.add_to_histogram(&mut hist[0], f1);
                                self.add_to_histogram(&mut hist[1], f2);
                                self.add_to_histogram(&mut hist[2], f3);
                            }
                        }
                    }
                }
            });
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, feat) in features.iter().enumerate() {
                #[cfg(feature = "progress")]
                pb.inc(1);

                let neighbors = self.radius_search(kdtree, &feat.point, self.fpfh_radius);

                for &n_idx in &neighbors {
                    let n_point = Vector3::new(
                        points.get(n_idx, 0),
                        points.get(n_idx, 1),
                        points.get(n_idx, 2),
                    );
                    let key = (
                        (n_point.x * 1e6) as i64,
                        (n_point.y * 1e6) as i64,
                        (n_point.z * 1e6) as i64,
                    );

                    if let Some(&n_feat_idx) = point_to_feature.get(&key) {
                        if i != n_feat_idx {
                            if let Some((f1, f2, f3)) = self.compute_pair_features(
                                &feat.point,
                                &feat.normal,
                                &features[n_feat_idx].point,
                                &features[n_feat_idx].normal,
                            ) {
                                self.add_to_histogram(&mut spfh_histograms[i][0], f1);
                                self.add_to_histogram(&mut spfh_histograms[i][1], f2);
                                self.add_to_histogram(&mut spfh_histograms[i][2], f3);
                            }
                        }
                    }
                }
            }
        }

        #[cfg(feature = "progress")]
        pb.finish();

        // Convert Mutex-wrapped histograms back to regular Vec for non-rayon compatibility
        #[cfg(feature = "rayon")]
        let spfh_histograms: Vec<[Vec<f64>; 3]> = spfh_histograms
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();

        // Compute FPFH by weighted averaging of SPFH
        #[cfg(feature = "progress")]
        let pb2 = crate::progress::create_progress_bar(features.len() as u64, "Computing FPFH");

        #[cfg(feature = "rayon")]
        {
            features.par_iter_mut().enumerate().for_each(|(i, feat)| {
                #[cfg(feature = "progress")]
                pb2.inc(1);

                let neighbors = self.radius_search(kdtree, &feat.point, self.fpfh_radius);

                let mut fpfh = vec![0.0; 3 * self.bins];
                let mut weight_sum = 0.0;

                for &n_idx in &neighbors {
                    let n_point = Vector3::new(
                        points.get(n_idx, 0),
                        points.get(n_idx, 1),
                        points.get(n_idx, 2),
                    );
                    let key = (
                        (n_point.x * 1e6) as i64,
                        (n_point.y * 1e6) as i64,
                        (n_point.z * 1e6) as i64,
                    );

                    if let Some(&n_feat_idx) = point_to_feature.get(&key) {
                        let dist = (feat.point - n_point).norm();
                        let weight = if dist > 1e-10 { 1.0 / dist } else { 0.0 };

                        for k in 0..self.bins {
                            fpfh[k] += weight * spfh_histograms[n_feat_idx][0][k];
                            fpfh[self.bins + k] += weight * spfh_histograms[n_feat_idx][1][k];
                            fpfh[2 * self.bins + k] += weight * spfh_histograms[n_feat_idx][2][k];
                        }
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    for val in &mut fpfh {
                        *val /= weight_sum;
                    }
                }

                feat.descriptor = fpfh;
            });
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, feat) in features.iter_mut().enumerate() {
                #[cfg(feature = "progress")]
                pb2.inc(1);

                let neighbors = self.radius_search(kdtree, &feat.point, self.fpfh_radius);

                let mut fpfh = vec![0.0; 3 * self.bins];
                let mut weight_sum = 0.0;

                for &n_idx in &neighbors {
                    let n_point = Vector3::new(
                        points.get(n_idx, 0),
                        points.get(n_idx, 1),
                        points.get(n_idx, 2),
                    );
                    let key = (
                        (n_point.x * 1e6) as i64,
                        (n_point.y * 1e6) as i64,
                        (n_point.z * 1e6) as i64,
                    );

                    if let Some(&n_feat_idx) = point_to_feature.get(&key) {
                        let dist = (feat.point - n_point).norm();
                        let weight = if dist > 1e-10 { 1.0 / dist } else { 0.0 };

                        for k in 0..self.bins {
                            fpfh[k] += weight * spfh_histograms[n_feat_idx][0][k];
                            fpfh[self.bins + k] += weight * spfh_histograms[n_feat_idx][1][k];
                            fpfh[2 * self.bins + k] += weight * spfh_histograms[n_feat_idx][2][k];
                        }
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    for val in &mut fpfh {
                        *val /= weight_sum;
                    }
                }

                feat.descriptor = fpfh;
            }
        }

        #[cfg(feature = "progress")]
        pb2.finish();
    }

    /// Compute angular features between two oriented points
    ///
    /// Returns (f1, f2, f3) as per PCL/FPFH definition
    fn compute_pair_features(
        &self,
        p1: &Vector3<f64>,
        n1: &Vector3<f64>,
        p2: &Vector3<f64>,
        n2: &Vector3<f64>,
    ) -> Option<(f64, f64, f64)> {
        let diff = p2 - p1;
        let dist = diff.norm();
        if dist < 1e-10 {
            return None;
        }

        let u = n1.normalize();
        let v = diff / dist;
        let w = u.cross(&v);

        // f1 = angle between n2 and u
        let f1 = n2.dot(&u);

        // f2 = angle between v and n2 projected on v-w plane
        let v_proj = v.dot(&n2);
        let w_proj = w.dot(&n2);
        let f2 = w_proj.atan2(v_proj);

        // f3 = angle between u and diff
        let f3 = u.dot(&v);

        Some((f1, f2, f3))
    }

    /// Add feature value to histogram bin
    fn add_to_histogram(&self, hist: &mut [f64], value: f64) {
        // Map value from [-1, 1] to [0, bins-1]
        let normalized = (value + 1.0) / 2.0; // [0, 1]
        let bin_idx = (normalized * (self.bins as f64 - 1.0)).round() as usize;
        let bin_idx = bin_idx.min(self.bins - 1);
        hist[bin_idx] += 1.0;
    }
}

impl Default for FasterPFH {
    fn default() -> Self {
        Self::new(0.9, 1.5, 1.0, 11)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fpfh_basic() {
        // Create simple point cloud
        let points = DataMatrix::from_row_slice(
            10,
            3,
            &[
                0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1,
                0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0, 0.05, 0.05, 0.1,
            ],
        );

        let fpfh = FasterPFH::new(0.15, 0.2, 1.0, 11);
        let features = fpfh.compute_features(&points);

        assert!(features.len() > 0, "Should extract some features");

        for feat in &features {
            assert!(feat.is_valid, "All features should be valid");
            assert_eq!(feat.descriptor.len(), 33, "FPFH should have 33 values");
            assert!(feat.normal.norm() > 0.0, "Normal should be non-zero");
        }

        println!(
            "Extracted {} features from {} points",
            features.len(),
            points.n_points()
        );
    }
}
