//! Feature-based correspondence matching for KISS-Matcher
//!
//! This module implements feature matching between two point clouds using
//! FPFH descriptors. It uses mutual nearest neighbors with Lowe's ratio test
//! to generate robust initial correspondences.
//!
//! # Algorithm
//!
//! 1. **Nearest Neighbor Search**: For each feature in source, find nearest in target
//! 2. **Ratio Test**: Reject ambiguous matches (Lowe's ratio test)
//! 3. **Mutual Matching**: Only keep bidirectional matches
//! 4. **Generate Correspondences**: Create 6D point pairs for registration
//!
//! # Example
//!
//! ```ignore
//! use inlier::kiss_matcher::correspondence::FeatureMatcher;
//! use inlier::kiss_matcher::features::FasterPFH;
//!
//! // Extract features from source and target
//! let fpfh = FasterPFH::new(0.2, 0.35, 0.9, 11);
//! let src_features = fpfh.compute_features(&src_points);
//! let tgt_features = fpfh.compute_features(&tgt_points);
//!
//! // Match features
//! let matcher = FeatureMatcher::new(0.8); // ratio threshold
//! let correspondences = matcher.match_features(&src_features, &tgt_features);
//! ```

use super::features::FeaturePoint;
use crate::types::DataMatrix;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Feature matcher using mutual nearest neighbors and ratio test
pub struct FeatureMatcher {
    /// Ratio test threshold (Lowe's ratio test, typically 0.6-0.8)
    pub ratio_threshold: f64,
    /// Use mutual matching (bidirectional check)
    pub use_mutual: bool,
}

/// A correspondence between two feature points
#[derive(Clone, Debug)]
pub struct Correspondence {
    pub src_idx: usize,
    pub tgt_idx: usize,
    pub distance: f64,
}

impl FeatureMatcher {
    /// Create new feature matcher
    ///
    /// # Arguments
    /// * `ratio_threshold` - Lowe's ratio test threshold (typically 0.6-0.8)
    ///   - Lower values are more restrictive (fewer but better matches)
    ///   - 0.8 is a good default balance
    pub fn new(ratio_threshold: f64) -> Self {
        Self {
            ratio_threshold,
            use_mutual: true,
        }
    }

    /// Match features between source and target point clouds
    ///
    /// Returns vector of correspondences (src_idx, tgt_idx, distance)
    pub fn match_features(
        &self,
        src_features: &[FeaturePoint],
        tgt_features: &[FeaturePoint],
    ) -> Vec<Correspondence> {
        if src_features.is_empty() || tgt_features.is_empty() {
            return Vec::new();
        }

        // Find matches from source to target
        let src_to_tgt = self.find_nearest_neighbors(src_features, tgt_features);

        if !self.use_mutual {
            return src_to_tgt;
        }

        // Find matches from target to source
        let tgt_to_src = self.find_nearest_neighbors(tgt_features, src_features);

        // Keep only mutual matches
        self.filter_mutual_matches(&src_to_tgt, &tgt_to_src)
    }

    /// Find nearest neighbors with ratio test
    fn find_nearest_neighbors(
        &self,
        query_features: &[FeaturePoint],
        search_features: &[FeaturePoint],
    ) -> Vec<Correspondence> {
        #[cfg(feature = "progress")]
        let pb = crate::progress::create_progress_bar(query_features.len() as u64, "Matching features");

        #[cfg(feature = "rayon")]
        let iter = query_features.par_iter().enumerate();
        #[cfg(not(feature = "rayon"))]
        let iter = query_features.iter().enumerate();

        let matches: Vec<_> = iter
            .filter_map(|(query_idx, query_feat)| {
                #[cfg(feature = "progress")]
                pb.inc(1);

                if !query_feat.is_valid {
                    return None;
                }

                // Find two nearest neighbors
                let (best_idx, best_dist, second_best_dist) =
                    self.find_two_nearest(&query_feat.descriptor, search_features);

                let best_idx = best_idx?;

                // Apply Lowe's ratio test: best_dist / second_best_dist < threshold
                if second_best_dist > 0.0 {
                    let ratio = best_dist / second_best_dist;
                    if ratio > self.ratio_threshold {
                        ; // Ambiguous match, reject
                        return None; // Ambiguous match, reject
                    }
                }

                Some(Correspondence {
                    src_idx: query_idx,
                    tgt_idx: best_idx,
                    distance: best_dist,
                })
            })
            .collect();

        #[cfg(feature = "progress")]
        pb.finish();

        matches
    }

    /// Find two nearest neighbors in descriptor space
    ///
    /// Returns (best_idx, best_dist, second_best_dist)
    fn find_two_nearest(
        &self,
        query_desc: &[f64],
        search_features: &[FeaturePoint],
    ) -> (Option<usize>, f64, f64) {
        let mut best_idx = None;
        let mut best_dist = f64::INFINITY;
        let mut second_best_dist = f64::INFINITY;

        for (idx, feat) in search_features.iter().enumerate() {
            if !feat.is_valid {
                continue;
            }

            let dist = self.descriptor_distance(query_desc, &feat.descriptor);

            if dist < best_dist {
                second_best_dist = best_dist;
                best_dist = dist;
                best_idx = Some(idx);
            } else if dist < second_best_dist {
                second_best_dist = dist;
            }
        }

        (best_idx, best_dist, second_best_dist)
    }

    /// Compute L2 distance between descriptors
    fn descriptor_distance(&self, desc1: &[f64], desc2: &[f64]) -> f64 {
        assert_eq!(desc1.len(), desc2.len(), "Descriptor dimensions must match");

        let mut sum = 0.0;
        for (a, b) in desc1.iter().zip(desc2.iter()) {
            let diff = a - b;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Filter to keep only mutual matches
    fn filter_mutual_matches(
        &self,
        src_to_tgt: &[Correspondence],
        tgt_to_src: &[Correspondence],
    ) -> Vec<Correspondence> {
        let mut mutual_matches = Vec::new();

        // Build reverse lookup: tgt_to_src maps target_idx -> source_idx
        // Note: In tgt_to_src, src_idx is actually a target index (since we queried from target)
        //       and tgt_idx is actually a source index
        let max_src_idx = tgt_to_src.iter().map(|c| c.src_idx).max().unwrap_or(0);

        let mut tgt_to_src_map = vec![None; max_src_idx + 1];

        for corr in tgt_to_src {
            // corr.src_idx is the target feature index (query)
            // corr.tgt_idx is the source feature index (match)
            if corr.src_idx < tgt_to_src_map.len() {
                tgt_to_src_map[corr.src_idx] = Some(corr.tgt_idx);
            }
        }

        // Check for mutual matches
        for corr in src_to_tgt {
            // corr.src_idx is source feature index
            // corr.tgt_idx is target feature index
            if let Some(&Some(reverse_match)) = tgt_to_src_map.get(corr.tgt_idx) {
                if reverse_match == corr.src_idx {
                    // Mutual match found
                    mutual_matches.push(corr.clone());
                }
            }
        }

        mutual_matches
    }

    /// Convert correspondences to DataMatrix format (6D: 3D source + 3D target)
    ///
    /// Returns a DataMatrix of shape (N, 6) where each row is [src_x, src_y, src_z, tgt_x, tgt_y, tgt_z]
    pub fn correspondences_to_matrix(
        &self,
        correspondences: &[Correspondence],
        src_features: &[FeaturePoint],
        tgt_features: &[FeaturePoint],
    ) -> DataMatrix {
        let n = correspondences.len();
        let mut data = Vec::with_capacity(n * 6);

        for corr in correspondences {
            let src = &src_features[corr.src_idx].point;
            let tgt = &tgt_features[corr.tgt_idx].point;

            data.push(src.x);
            data.push(src.y);
            data.push(src.z);
            data.push(tgt.x);
            data.push(tgt.y);
            data.push(tgt.z);
        }

        DataMatrix::from_row_slice(n, 6, &data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_exact_match() {
        // Create identical features
        let src = vec![
            FeaturePoint {
                point: Vector3::new(0.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![1.0, 0.5, 0.3],
                is_valid: true,
            },
            FeaturePoint {
                point: Vector3::new(1.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.8, 0.6, 0.2],
                is_valid: true,
            },
        ];

        let tgt = src.clone();

        let matcher = FeatureMatcher::new(0.8);
        let matches = matcher.match_features(&src, &tgt);

        // Should find 2 mutual matches (exact duplicates)
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].src_idx, 0);
        assert_eq!(matches[0].tgt_idx, 0);
        assert!(matches[0].distance < 1e-10);
    }

    #[test]
    fn test_ratio_test_filtering() {
        // Create source with one feature
        let src = vec![FeaturePoint {
            point: Vector3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
            descriptor: vec![1.0, 0.0, 0.0],
            is_valid: true,
        }];

        // Create targets with two very similar features (ambiguous)
        let tgt = vec![
            FeaturePoint {
                point: Vector3::new(0.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.95, 0.0, 0.0], // Very close to source
                is_valid: true,
            },
            FeaturePoint {
                point: Vector3::new(1.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.94, 0.0, 0.0], // Also very close to source
                is_valid: true,
            },
        ];

        // With strict ratio threshold, ambiguous match should be rejected
        let matcher = FeatureMatcher::new(0.7);
        let matches = matcher.match_features(&src, &tgt);

        // Ratio = 0.05 / 0.06 ≈ 0.83 > 0.7, should reject
        assert_eq!(matches.len(), 0, "Ambiguous match should be rejected");
    }

    #[test]
    fn test_mutual_matching() {
        let src = vec![
            FeaturePoint {
                point: Vector3::new(0.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![1.0, 0.0, 0.0],
                is_valid: true,
            },
            FeaturePoint {
                point: Vector3::new(1.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.0, 1.0, 0.0],
                is_valid: true,
            },
        ];

        // Target with swapped features
        let tgt = vec![
            FeaturePoint {
                point: Vector3::new(1.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.0, 1.0, 0.0],
                is_valid: true,
            },
            FeaturePoint {
                point: Vector3::new(0.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![1.0, 0.0, 0.0],
                is_valid: true,
            },
        ];

        let matcher = FeatureMatcher::new(0.8);
        let matches = matcher.match_features(&src, &tgt);

        // Should find 2 mutual matches
        assert_eq!(matches.len(), 2);

        // Check correct mapping: src[0] -> tgt[1], src[1] -> tgt[0]
        let match0 = matches.iter().find(|m| m.src_idx == 0).unwrap();
        let match1 = matches.iter().find(|m| m.src_idx == 1).unwrap();

        assert_eq!(match0.tgt_idx, 1);
        assert_eq!(match1.tgt_idx, 0);
    }

    #[test]
    fn test_correspondences_to_matrix() {
        let src = vec![
            FeaturePoint {
                point: Vector3::new(0.0, 0.0, 0.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![1.0],
                is_valid: true,
            },
            FeaturePoint {
                point: Vector3::new(1.0, 2.0, 3.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.5],
                is_valid: true,
            },
        ];

        let tgt = vec![
            FeaturePoint {
                point: Vector3::new(4.0, 5.0, 6.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![1.0],
                is_valid: true,
            },
            FeaturePoint {
                point: Vector3::new(7.0, 8.0, 9.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                descriptor: vec![0.5],
                is_valid: true,
            },
        ];

        let correspondences = vec![
            Correspondence {
                src_idx: 0,
                tgt_idx: 0,
                distance: 0.0,
            },
            Correspondence {
                src_idx: 1,
                tgt_idx: 1,
                distance: 0.0,
            },
        ];

        let matcher = FeatureMatcher::new(0.8);
        let matrix = matcher.correspondences_to_matrix(&correspondences, &src, &tgt);

        assert_eq!(matrix.n_points(), 2);
        assert_eq!(matrix.n_dims(), 6);

        // Check first correspondence: [0,0,0, 4,5,6]
        assert_eq!(matrix.get(0, 0), 0.0);
        assert_eq!(matrix.get(0, 1), 0.0);
        assert_eq!(matrix.get(0, 2), 0.0);
        assert_eq!(matrix.get(0, 3), 4.0);
        assert_eq!(matrix.get(0, 4), 5.0);
        assert_eq!(matrix.get(0, 5), 6.0);

        // Check second correspondence: [1,2,3, 7,8,9]
        assert_eq!(matrix.get(1, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 2.0);
        assert_eq!(matrix.get(1, 2), 3.0);
        assert_eq!(matrix.get(1, 3), 7.0);
        assert_eq!(matrix.get(1, 4), 8.0);
        assert_eq!(matrix.get(1, 5), 9.0);
    }
}
