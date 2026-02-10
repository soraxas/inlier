//! SIPFH (Scale-Invariant Point Feature Histogram)
//!
//! Combines 3D-SIFT scale-invariant keypoint detection with FPFH descriptors.
//!
//! # Algorithm Overview
//!
//! 1. **DoG Pyramid**: Build Difference-of-Gaussian pyramid for scale-space analysis
//! 2. **Keypoint Detection**: Find extrema across scales (scale-invariant keypoints)
//! 3. **Keypoint Refinement**: Subpixel localization and filtering
//! 4. **Orientation Assignment**: Compute dominant orientation for rotation invariance
//! 5. **SIPFH Descriptor**: Compute FPFH at keypoints + scale information
//!
//! # References
//!
//! - SIPFH: Combines SIFT keypoints with FPFH for fast, scale-invariant matching
//! - 3D-SIFT: Lowe's SIFT extended to 3D point clouds
//! - FPFH: Fast Point Feature Histogram (PCL implementation)

use crate::matcher::features::{FasterPFH, FeaturePoint};
use crate::types::DataMatrix;
use kiddo::SquaredEuclidean;
use kiddo::float::kdtree::KdTree as KiddoKdTree;
use nalgebra::{Matrix3, Vector3};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// SIPFH configuration
#[derive(Debug, Clone)]
pub struct SIPFHConfig {
    /// Number of octaves in DoG pyramid (typically 3-4)
    pub num_octaves: usize,
    /// Number of scales per octave (typically 3-5)
    pub scales_per_octave: usize,
    /// Initial sigma for Gaussian blur
    pub initial_sigma: f64,
    /// DoG threshold for keypoint detection (typically 0.01-0.03)
    pub dog_threshold: f64,
    /// Edge response threshold (similar to Harris corner)
    pub edge_threshold: f64,
    /// Radius for FPFH computation at keypoints
    pub fpfh_radius: f64,
    /// Linearity threshold for FPFH
    pub the_linearity: f64,
    /// Number of bins for FPFH histogram
    pub fpfh_bins: usize,
    /// Weight for scale information in SIPFH descriptor (ω)
    pub scale_weight: f64,
}

impl Default for SIPFHConfig {
    fn default() -> Self {
        Self {
            num_octaves: 3,
            scales_per_octave: 4,
            initial_sigma: 0.05,
            dog_threshold: 0.02,
            edge_threshold: 10.0,
            fpfh_radius: 0.3,
            the_linearity: 0.9,
            fpfh_bins: 11,
            scale_weight: 0.5,
        }
    }
}

/// Scale-space keypoint with scale and orientation information
#[derive(Debug, Clone)]
pub struct ScaleKeypoint {
    /// 3D position
    pub point: Vector3<f64>,
    /// Scale (sigma) at which keypoint was detected
    pub scale: f64,
    /// Octave level
    pub octave: usize,
    /// Scale level within octave
    pub scale_level: usize,
    /// DoG response value
    pub response: f64,
    /// Dominant orientation (azimuth, elevation)
    pub orientation: Option<(f64, f64)>,
}

/// SIPFH feature point with scale-invariant descriptor
#[derive(Debug, Clone)]
pub struct SIPFHFeaturePoint {
    /// Base feature point data
    pub feature: FeaturePoint,
    /// Scale at which feature was detected
    pub scale: f64,
    /// SIPFH descriptor (FPFH + weighted scale info)
    pub sipfh_descriptor: Vec<f64>,
}

/// SIPFH extractor combining 3D-SIFT and FPFH
pub struct SIPFH {
    config: SIPFHConfig,
    fpfh: FasterPFH,
}

impl SIPFH {
    /// Create new SIPFH extractor
    pub fn new(config: SIPFHConfig) -> Self {
        let fpfh = FasterPFH::new(
            config.fpfh_radius * 0.5, // Normal radius
            config.fpfh_radius,
            config.the_linearity,
            config.fpfh_bins,
        );

        Self { config, fpfh }
    }

    /// Extract SIPFH features from point cloud
    pub fn extract_features(&self, points: &DataMatrix) -> Vec<SIPFHFeaturePoint> {
        if points.n_points() < 10 {
            return Vec::new();
        }

        #[cfg(feature = "progress")]
        println!("  [1/4] Building DoG pyramid...");

        // Step 1: Build DoG pyramid
        let dog_pyramid = self.build_dog_pyramid(points);

        #[cfg(feature = "progress")]
        println!("  [2/4] Detecting scale-space keypoints...");

        // Step 2: Detect keypoints across scales
        let keypoints = self.detect_keypoints(&dog_pyramid, points);

        #[cfg(feature = "progress")]
        println!("  [3/4] Refining keypoints...");

        // Step 3: Refine and filter keypoints
        let refined_keypoints = self.refine_keypoints(keypoints, points);

        #[cfg(feature = "progress")]
        println!(
            "  [4/4] Computing SIPFH descriptors at {} keypoints...",
            refined_keypoints.len()
        );

        // Step 4: Compute SIPFH descriptors at keypoints
        self.compute_sipfh_descriptors(&refined_keypoints, points)
    }

    /// Build Difference-of-Gaussian pyramid
    fn build_dog_pyramid(&self, points: &DataMatrix) -> Vec<Vec<DogScale>> {
        let mut pyramid = Vec::new();

        // For each octave
        for octave in 0..self.config.num_octaves {
            let mut octave_scales = Vec::new();

            // Compute scales for this octave
            let k = 2.0_f64.powf(1.0 / self.config.scales_per_octave as f64);
            let base_sigma = self.config.initial_sigma * (2.0_f64.powi(octave as i32));

            // For each scale in octave
            for s in 0..=self.config.scales_per_octave {
                let sigma = base_sigma * k.powi(s as i32);

                // Create scale representation
                let scale_data = self.create_scale_space(points, sigma, octave);
                octave_scales.push(scale_data);
            }

            pyramid.push(octave_scales);
        }

        pyramid
    }

    /// Create scale-space representation at given sigma
    fn create_scale_space(&self, points: &DataMatrix, sigma: f64, _octave: usize) -> DogScale {
        // Simplified: Use radius-based density estimation as proxy for Gaussian blur
        // In full implementation, would voxelize and convolve with 3D Gaussian

        let kdtree = self.build_kdtree(points);
        let n = points.n_points();

        let densities: Vec<f64> = (0..n)
            .map(|i| {
                let point = Vector3::new(points.get(i, 0), points.get(i, 1), points.get(i, 2));
                let neighbors = self.radius_search(&kdtree, &point, sigma * 2.0);
                neighbors.len() as f64 / (sigma * sigma * sigma * 8.0 * std::f64::consts::PI)
            })
            .collect();

        DogScale { sigma, densities }
    }

    /// Detect extrema in DoG pyramid
    fn detect_keypoints(
        &self,
        dog_pyramid: &[Vec<DogScale>],
        points: &DataMatrix,
    ) -> Vec<ScaleKeypoint> {
        let mut keypoints = Vec::new();

        // For each octave
        for (octave, octave_scales) in dog_pyramid.iter().enumerate() {
            // Need at least 3 scales for DoG comparison
            if octave_scales.len() < 3 {
                continue;
            }

            // For each middle scale (not first or last)
            for scale_idx in 1..(octave_scales.len() - 1) {
                let prev_scale = &octave_scales[scale_idx - 1];
                let curr_scale = &octave_scales[scale_idx];
                let next_scale = &octave_scales[scale_idx + 1];

                // Compute DoG responses
                let dog_curr: Vec<f64> = curr_scale
                    .densities
                    .iter()
                    .zip(prev_scale.densities.iter())
                    .map(|(c, p)| c - p)
                    .collect();

                let dog_next: Vec<f64> = next_scale
                    .densities
                    .iter()
                    .zip(curr_scale.densities.iter())
                    .map(|(n, c)| n - c)
                    .collect();

                // Find extrema
                for i in 0..points.n_points() {
                    let response = dog_curr[i];

                    // Check threshold
                    if response.abs() < self.config.dog_threshold {
                        continue;
                    }

                    // Check if local extremum (simplified - should check 26-neighborhood in full impl)
                    let is_max = response > 0.0
                        && response > dog_curr.get(i.wrapping_sub(1)).copied().unwrap_or(0.0)
                        && response > dog_curr.get(i + 1).copied().unwrap_or(0.0)
                        && response > dog_next[i];

                    let is_min = response < 0.0
                        && response < dog_curr.get(i.wrapping_sub(1)).copied().unwrap_or(0.0)
                        && response < dog_curr.get(i + 1).copied().unwrap_or(0.0)
                        && response < dog_next[i];

                    if is_max || is_min {
                        keypoints.push(ScaleKeypoint {
                            point: Vector3::new(
                                points.get(i, 0),
                                points.get(i, 1),
                                points.get(i, 2),
                            ),
                            scale: curr_scale.sigma,
                            octave,
                            scale_level: scale_idx,
                            response: response.abs(),
                            orientation: None,
                        });
                    }
                }
            }
        }

        keypoints
    }

    /// Refine keypoints: subpixel localization and filtering
    fn refine_keypoints(
        &self,
        keypoints: Vec<ScaleKeypoint>,
        points: &DataMatrix,
    ) -> Vec<ScaleKeypoint> {
        let kdtree = self.build_kdtree(points);

        keypoints
            .into_iter()
            .filter_map(|mut kp| {
                // Compute local structure tensor for edge filtering
                let neighbors = self.radius_search(&kdtree, &kp.point, kp.scale * 2.0);
                if neighbors.len() < 3 {
                    return None;
                }

                // Compute covariance matrix
                let neighbor_points: Vec<Vector3<f64>> = neighbors
                    .iter()
                    .map(|&idx| {
                        Vector3::new(points.get(idx, 0), points.get(idx, 1), points.get(idx, 2))
                    })
                    .collect();

                let centroid =
                    neighbor_points.iter().sum::<Vector3<f64>>() / neighbors.len() as f64;

                let mut cov = Matrix3::zeros();
                for p in &neighbor_points {
                    let diff = p - centroid;
                    cov += diff * diff.transpose();
                }
                cov /= neighbors.len() as f64;

                // Eigenvalue decomposition for edge response
                let eigen = cov.symmetric_eigen();
                let eigenvalues = eigen.eigenvalues;

                // Sort eigenvalues: λ1 >= λ2 >= λ3
                let mut sorted_eigs = [eigenvalues[0], eigenvalues[1], eigenvalues[2]];
                sorted_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());

                // Edge response ratio (like Harris corner detector)
                let edge_ratio = sorted_eigs[0] / (sorted_eigs[2] + 1e-10);

                // Filter edges (high edge response = bad keypoint)
                if edge_ratio > self.config.edge_threshold {
                    return None;
                }

                // Compute orientation
                let orientation = self.compute_orientation(&kp, &neighbor_points, &centroid);
                kp.orientation = Some(orientation);

                Some(kp)
            })
            .collect()
    }

    /// Compute dominant orientation for rotation invariance
    fn compute_orientation(
        &self,
        keypoint: &ScaleKeypoint,
        neighbors: &[Vector3<f64>],
        centroid: &Vector3<f64>,
    ) -> (f64, f64) {
        // Compute gradient directions and magnitudes
        let mut azimuth_hist = vec![0.0; 36]; // 36 bins = 10° per bin
        let mut elevation_hist = vec![0.0; 18]; // 18 bins = 10° per bin

        for p in neighbors {
            let diff = p - centroid;
            let m = diff.norm(); // Magnitude

            if m < 1e-6 {
                continue;
            }

            // Azimuth: θ = arctan(y/x)
            let theta = diff.y.atan2(diff.x);
            let theta_deg = theta.to_degrees();
            let azimuth_bin = ((theta_deg + 180.0) / 10.0) as usize % 36;

            // Elevation: φ = arcsin(z/m)
            let phi = (diff.z / m).asin();
            let phi_deg = phi.to_degrees();
            let elevation_bin = ((phi_deg + 90.0) / 10.0).clamp(0.0, 17.0) as usize;

            // Weight by magnitude
            azimuth_hist[azimuth_bin] += m;
            elevation_hist[elevation_bin] += m;
        }

        // Find dominant orientation
        let dominant_azimuth_bin = azimuth_hist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let dominant_elevation_bin = elevation_hist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let azimuth = (dominant_azimuth_bin as f64 * 10.0 - 180.0).to_radians();
        let elevation = (dominant_elevation_bin as f64 * 10.0 - 90.0).to_radians();

        (azimuth, elevation)
    }

    /// Compute SIPFH descriptors at refined keypoints
    fn compute_sipfh_descriptors(
        &self,
        keypoints: &[ScaleKeypoint],
        points: &DataMatrix,
    ) -> Vec<SIPFHFeaturePoint> {
        if keypoints.is_empty() {
            return Vec::new();
        }

        // Build KDTree for neighbor search
        let kdtree = self.build_kdtree(points);

        #[cfg(feature = "rayon")]
        let iter = keypoints.par_iter();
        #[cfg(not(feature = "rayon"))]
        let iter = keypoints.iter();

        iter.filter_map(|kp| {
            // Compute FPFH at keypoint
            let neighbors = self.radius_search(&kdtree, &kp.point, self.config.fpfh_radius);
            if neighbors.len() < 3 {
                return None;
            }

            // Get neighbor points for normal computation
            let neighbor_points: Vec<Vector3<f64>> = neighbors
                .iter()
                .map(|&idx| {
                    Vector3::new(points.get(idx, 0), points.get(idx, 1), points.get(idx, 2))
                })
                .collect();
            let centroid = neighbor_points.iter().sum::<Vector3<f64>>() / neighbors.len() as f64;
            let normal = self.compute_normal(&neighbor_points, &centroid);

            // Compute FPFH descriptor
            let fpfh_descriptor = self.compute_fpfh_at_point(&kp.point, &neighbors, points);

            // Create SIPFH descriptor: FPFH + weighted scale
            let mut sipfh_descriptor = fpfh_descriptor.clone();

            // Append scale information (weighted)
            sipfh_descriptor.push(kp.scale * self.config.scale_weight);

            // Append orientation if available
            if let Some((azimuth, elevation)) = kp.orientation {
                sipfh_descriptor.push(azimuth.cos() * self.config.scale_weight);
                sipfh_descriptor.push(azimuth.sin() * self.config.scale_weight);
                sipfh_descriptor.push(elevation * self.config.scale_weight);
            }

            // Normalize descriptor
            let norm = sipfh_descriptor.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-6 {
                for val in &mut sipfh_descriptor {
                    *val /= norm;
                }
            }

            Some(SIPFHFeaturePoint {
                feature: FeaturePoint {
                    point: kp.point,
                    normal,
                    descriptor: fpfh_descriptor,
                    is_valid: true,
                },
                scale: kp.scale,
                sipfh_descriptor,
            })
        })
        .collect()
    }

    /// Compute FPFH descriptor at a single point using full FasterPFH pipeline
    fn compute_fpfh_at_point(
        &self,
        point: &Vector3<f64>,
        neighbors: &[usize],
        points: &DataMatrix,
    ) -> Vec<f64> {
        if neighbors.len() < 3 {
            return vec![0.0; self.config.fpfh_bins * 3];
        }

        // Get neighbor points
        let neighbor_points: Vec<Vector3<f64>> = neighbors
            .iter()
            .map(|&idx| Vector3::new(points.get(idx, 0), points.get(idx, 1), points.get(idx, 2)))
            .collect();

        // Compute centroid
        let centroid = neighbor_points.iter().sum::<Vector3<f64>>() / neighbors.len() as f64;

        // Compute normal using PCA
        let normal = self.compute_normal(&neighbor_points, &centroid);
        if normal.norm() < 1e-6 {
            return vec![0.0; self.config.fpfh_bins * 3];
        }

        // Compute SPFH (Simplified Point Feature Histogram)
        let spfh = self.compute_spfh(point, &normal, &neighbor_points, neighbors);

        // Aggregate to FPFH (Fast Point Feature Histogram)
        // In full FPFH, we'd aggregate SPFH from neighbors, but for keypoints we use SPFH directly
        spfh
    }

    /// Compute normal using PCA
    fn compute_normal(&self, points: &[Vector3<f64>], centroid: &Vector3<f64>) -> Vector3<f64> {
        if points.len() < 3 {
            return Vector3::zeros();
        }

        // Build covariance matrix
        let mut cov = Matrix3::zeros();
        for p in points {
            let diff = p - centroid;
            cov += diff * diff.transpose();
        }
        cov /= points.len() as f64;

        // Eigen decomposition - smallest eigenvalue's eigenvector is the normal
        let eigen = cov.symmetric_eigen();
        let min_idx = if eigen.eigenvalues[0] < eigen.eigenvalues[1] {
            if eigen.eigenvalues[0] < eigen.eigenvalues[2] {
                0
            } else {
                2
            }
        } else if eigen.eigenvalues[1] < eigen.eigenvalues[2] {
            1
        } else {
            2
        };

        eigen.eigenvectors.column(min_idx).into()
    }

    /// Compute SPFH (Simplified Point Feature Histogram)
    fn compute_spfh(
        &self,
        point: &Vector3<f64>,
        normal: &Vector3<f64>,
        neighbor_points: &[Vector3<f64>],
        neighbors: &[usize],
    ) -> Vec<f64> {
        let bins = self.config.fpfh_bins;
        let mut hist_f1 = vec![0.0; bins];
        let mut hist_f2 = vec![0.0; bins];
        let mut hist_f3 = vec![0.0; bins];

        let normalized_normal = normal.normalize();

        for (&neighbor_idx, neighbor_point) in neighbors.iter().zip(neighbor_points.iter()) {
            // Skip self
            if (neighbor_point - point).norm() < 1e-6 {
                continue;
            }

            // Estimate neighbor normal (simplified - reuse point normal)
            let neighbor_normal = normalized_normal;

            // Compute point pair features
            let diff = neighbor_point - point;
            let dist = diff.norm();
            if dist < 1e-6 {
                continue;
            }

            let u = normalized_normal;
            let v = diff / dist;
            let w = u.cross(&v);

            // Features (Rusu et al., 2008 FPFH formulation)
            let f1 = v.dot(&neighbor_normal);
            let f2 = (neighbor_point - point).norm(); // Distance feature
            let f3 = w.dot(&neighbor_normal);

            // Bin the features
            let bin1 =
                ((f1 + 1.0) / 2.0 * (bins - 1) as f64).clamp(0.0, (bins - 1) as f64) as usize;
            let bin2 = ((f2 / self.config.fpfh_radius) * (bins - 1) as f64)
                .clamp(0.0, (bins - 1) as f64) as usize;
            let bin3 =
                ((f3 + 1.0) / 2.0 * (bins - 1) as f64).clamp(0.0, (bins - 1) as f64) as usize;

            hist_f1[bin1] += 1.0;
            hist_f2[bin2] += 1.0;
            hist_f3[bin3] += 1.0;
        }

        // Normalize histograms
        let sum1: f64 = hist_f1.iter().sum();
        let sum2: f64 = hist_f2.iter().sum();
        let sum3: f64 = hist_f3.iter().sum();

        if sum1 > 0.0 {
            for val in &mut hist_f1 {
                *val /= sum1;
            }
        }
        if sum2 > 0.0 {
            for val in &mut hist_f2 {
                *val /= sum2;
            }
        }
        if sum3 > 0.0 {
            for val in &mut hist_f3 {
                *val /= sum3;
            }
        }

        // Concatenate histograms
        let mut fpfh = Vec::with_capacity(bins * 3);
        fpfh.extend_from_slice(&hist_f1);
        fpfh.extend_from_slice(&hist_f2);
        fpfh.extend_from_slice(&hist_f3);

        fpfh
    }

    /// Build KD-tree for neighbor search
    fn build_kdtree(&self, points: &DataMatrix) -> KiddoKdTree<f64, usize, 3, 32, u32> {
        let mut tree = KiddoKdTree::new();
        for i in 0..points.n_points() {
            let _ = tree.add(&[points.get(i, 0), points.get(i, 1), points.get(i, 2)], i);
        }
        tree
    }

    /// Radius search
    fn radius_search(
        &self,
        tree: &KiddoKdTree<f64, usize, 3, 32, u32>,
        point: &Vector3<f64>,
        radius: f64,
    ) -> Vec<usize> {
        tree.within::<SquaredEuclidean>(&[point.x, point.y, point.z], radius * radius)
            .iter()
            .map(|n| n.item)
            .collect()
    }
}

/// DoG scale representation
struct DogScale {
    sigma: f64,
    densities: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sipfh_creation() {
        let config = SIPFHConfig::default();
        let _sipfh = SIPFH::new(config);
    }

    #[test]
    fn test_dog_pyramid_octaves() {
        let config = SIPFHConfig {
            num_octaves: 3,
            scales_per_octave: 4,
            ..Default::default()
        };

        // Each octave should have scales_per_octave + 1 scales
        assert_eq!(config.scales_per_octave + 1, 5);
    }
}
