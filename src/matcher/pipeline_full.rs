//! KISS-Matcher pipeline integrating all components

use crate::matcher::config::KISSMatcherConfig;
use crate::matcher::correspondence::FeatureMatcher;
use crate::matcher::features::FasterPFH;
use crate::matcher::gnc::GNCSolver;
use crate::matcher::matching::ROBINMatching;
use crate::types::DataMatrix;
use nalgebra::{Matrix3, Vector3};

/// Result from full KISS-Matcher pipeline
#[derive(Clone, Debug)]
pub struct KISSMatcherFullResult {
    pub scale: f64,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub inlier_indices: Vec<usize>,
    pub n_correspondences_initial: usize,
    pub n_correspondences_after_robin: usize,
    pub n_correspondences_final: usize,
}

/// Run full KISS-Matcher pipeline for point cloud registration
///
/// # Arguments
/// * `src_points` - Source point cloud (N×3)
/// * `dst_points` - Destination point cloud (M×3)
/// * `config` - KISS-Matcher configuration
///
/// # Returns
/// * Complete transformation (scale, rotation, translation) and inliers
///
/// # Pipeline
/// 1. Voxel downsampling (if needed for dense clouds)
/// 2. Feature extraction (FasterPFH)
/// 3. Feature matching with ratio test
/// 4. Correspondence matrix construction (6D)
/// 5. Scale estimation using TLS on correspondences
/// 6. ROBIN k-core pruning
/// 7. GNC robust rotation/translation estimation
pub fn kiss_matcher_full_pipeline(
    src_points: &DataMatrix,
    dst_points: &DataMatrix,
    config: &KISSMatcherConfig,
) -> Option<KISSMatcherFullResult> {
    use crate::preprocessing::voxel_downsample;

    if src_points.n_points() < 3 || dst_points.n_points() < 3 {
        return None;
    }

    println!("\n=== KISS-Matcher Full Pipeline ===");
    println!(
        "Source: {} points, Target: {} points",
        src_points.n_points(),
        dst_points.n_points()
    );

    // Step 0: Downsample if clouds are too dense (>50k points)
    let (src_ds, dst_ds) = if src_points.n_points() > 50000 || dst_points.n_points() > 50000 {
        println!("\n[0/6] Downsampling dense point clouds...");
        let src_downsampled = voxel_downsample(src_points, config.voxel_size);
        let dst_downsampled = voxel_downsample(dst_points, config.voxel_size);
        println!(
            "  Downsampled to {} source, {} target points",
            src_downsampled.n_points(),
            dst_downsampled.n_points()
        );
        (src_downsampled, dst_downsampled)
    } else {
        (src_points.clone(), dst_points.clone())
    };

    // Step 1: Extract features using FasterPFH
    println!("\n[1/6] Extracting FasterPFH features...");
    let fpfh = FasterPFH::new(
        config.normal_radius,
        config.fpfh_radius,
        config.the_linearity,
        11, // bins
    );

    let src_features = fpfh.compute_features(&src_ds);
    let dst_features = fpfh.compute_features(&dst_ds);

    println!(
        "  Extracted {} source features, {} target features",
        src_features.len(),
        dst_features.len()
    );

    if src_features.is_empty() || dst_features.is_empty() {
        println!("  ERROR: No features extracted");
        return None;
    }

    // Step 2: Match features with ratio test
    println!("\n[2/6] Matching features...");
    let matcher = FeatureMatcher::new(config.ratio_threshold);
    let correspondences = matcher.match_features(&src_features, &dst_features);

    println!("  Found {} initial correspondences", correspondences.len());

    if correspondences.len() < 3 {
        println!("  ERROR: Too few correspondences");
        return None;
    }

    let n_initial = correspondences.len();

    // Convert to 6D matrix format [src_x, src_y, src_z, dst_x, dst_y, dst_z]
    let corr_matrix =
        matcher.correspondences_to_matrix(&correspondences, &src_features, &dst_features);

    // Step 3: Estimate scale using TLS on TIMs from correspondences
    println!("\n[3/6] Estimating scale with TLS...");
    let scale = estimate_scale_from_correspondences(&corr_matrix, config.solver_noise_bound)?;
    println!("  Estimated scale: {:.6}", scale);

    // Step 4: ROBIN k-core pruning on correspondences
    println!("\n[4/6] ROBIN k-core pruning...");
    let inlier_indices = robin_prune_correspondences(&corr_matrix, scale, config.robin_noise_bound);

    println!(
        "  Retained {} inliers ({:.1}%)",
        inlier_indices.len(),
        100.0 * inlier_indices.len() as f64 / n_initial as f64
    );

    if inlier_indices.len() < 3 {
        println!("  ERROR: Too few inliers after pruning");
        return None;
    }

    let n_after_robin = inlier_indices.len();

    // Extract inlier correspondences
    let inlier_corr = extract_correspondences(&corr_matrix, &inlier_indices);

    // Step 5: GNC solver for rotation and translation
    println!("\n[5/6] Solving for rotation and translation with GNC...");
    let gnc = GNCSolver::new(config.solver_noise_bound);
    let gnc_result = gnc.solve(&inlier_corr, scale);

    println!("  GNC iterations: {}", gnc_result.iterations);
    println!(
        "  Final inliers: {} ({:.1}%)",
        gnc_result.inliers.len(),
        100.0 * gnc_result.inliers.len() as f64 / n_after_robin as f64
    );

    println!("\n=== Results ===");
    println!("Scale: {:.6}", scale);
    println!("Rotation:\n{}", gnc_result.rotation);
    println!("Translation: {:?}", gnc_result.translation);
    println!(
        "Total inliers: {}/{} ({:.1}%)",
        gnc_result.inliers.len(),
        n_initial,
        100.0 * gnc_result.inliers.len() as f64 / n_initial as f64
    );

    let n_final = gnc_result.inliers.len();

    Some(KISSMatcherFullResult {
        scale,
        rotation: gnc_result.rotation,
        translation: gnc_result.translation,
        inlier_indices: gnc_result.inliers,
        n_correspondences_initial: n_initial,
        n_correspondences_after_robin: n_after_robin,
        n_correspondences_final: n_final,
    })
}

/// Helper: Estimate scale from 6D correspondence matrix using TLS
fn estimate_scale_from_correspondences(
    correspondences: &DataMatrix,
    noise_bound: f64,
) -> Option<f64> {
    use crate::estimators::tls::ScalarTLSEstimator;

    let n = correspondences.n_points();
    if n < 2 {
        return None;
    }

    // Compute TIMs (Translation Invariant Measurements) from correspondences
    let mut src_tims = Vec::new();
    let mut dst_tims = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            // Source TIM: p_j - p_i
            let src_tim = Vector3::new(
                correspondences.get(j, 0) - correspondences.get(i, 0),
                correspondences.get(j, 1) - correspondences.get(i, 1),
                correspondences.get(j, 2) - correspondences.get(i, 2),
            );

            // Destination TIM: q_j - q_i
            let dst_tim = Vector3::new(
                correspondences.get(j, 3) - correspondences.get(i, 3),
                correspondences.get(j, 4) - correspondences.get(i, 4),
                correspondences.get(j, 5) - correspondences.get(i, 5),
            );

            src_tims.push(src_tim);
            dst_tims.push(dst_tim);
        }
    }

    // Compute scale ratios
    let mut scale_measurements = Vec::new();
    let mut ranges = Vec::new();

    for (src_tim, dst_tim) in src_tims.iter().zip(dst_tims.iter()) {
        let src_norm = src_tim.norm();
        let dst_norm = dst_tim.norm();

        if src_norm > 1e-6 {
            let scale_ratio = dst_norm / src_norm;
            scale_measurements.push(scale_ratio);

            // Range based on noise propagation
            let range = 2.0 * noise_bound / src_norm;
            ranges.push(range);
        }
    }

    if scale_measurements.is_empty() {
        return None;
    }

    // Run TLS estimator
    let estimator = ScalarTLSEstimator::new(); // Uses default c_bar = 1.0
    let result = estimator.estimate(&scale_measurements, &ranges)?;

    Some(result.0) // Return scale, ignore inliers
}

/// Helper: ROBIN pruning on 6D correspondence matrix
fn robin_prune_correspondences(
    correspondences: &DataMatrix,
    scale: f64,
    noise_bound: f64,
) -> Vec<usize> {
    let n = correspondences.n_points();
    if n == 0 {
        return Vec::new();
    }

    // Build compatibility graph
    let mut graph = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            // Extract points
            let src_i = Vector3::new(
                correspondences.get(i, 0),
                correspondences.get(i, 1),
                correspondences.get(i, 2),
            );
            let dst_i = Vector3::new(
                correspondences.get(i, 3),
                correspondences.get(i, 4),
                correspondences.get(i, 5),
            );

            let src_j = Vector3::new(
                correspondences.get(j, 0),
                correspondences.get(j, 1),
                correspondences.get(j, 2),
            );
            let dst_j = Vector3::new(
                correspondences.get(j, 3),
                correspondences.get(j, 4),
                correspondences.get(j, 5),
            );

            // Geometric consistency check
            let src_dist = (src_j - src_i).norm();
            let dst_dist = (dst_j - dst_i).norm();

            let residual = (dst_dist - scale * src_dist).abs();

            if residual <= 2.0 * noise_bound {
                graph[i].push(j);
                graph[j].push(i);
            }
        }
    }

    // K-core decomposition (max_core mode)
    max_core_pruning(&graph)
}

/// Helper: K-core decomposition to find maximum k-core
fn max_core_pruning(graph: &[Vec<usize>]) -> Vec<usize> {
    let n = graph.len();
    let mut degrees: Vec<usize> = graph.iter().map(|neighbors| neighbors.len()).collect();
    let mut active = vec![true; n];

    // Find maximum k where k-core exists
    let mut k = 0;
    loop {
        let mut changed = false;

        // Remove nodes with degree < k
        for i in 0..n {
            if active[i] && degrees[i] < k {
                active[i] = false;
                changed = true;

                // Update neighbors' degrees
                for &j in &graph[i] {
                    if active[j] {
                        degrees[j] = degrees[j].saturating_sub(1);
                    }
                }
            }
        }

        if !changed {
            // Stable k-core found, try to increase k
            let active_count = active.iter().filter(|&&a| a).count();
            if active_count == 0 {
                break;
            }

            let min_degree = (0..n)
                .filter(|&i| active[i])
                .map(|i| degrees[i])
                .min()
                .unwrap_or(0);

            if min_degree <= k {
                break;
            }

            k += 1;
        }
    }

    // Return active nodes
    (0..n).filter(|&i| active[i]).collect()
}

/// Helper: Extract subset of correspondences by indices
fn extract_correspondences(correspondences: &DataMatrix, indices: &[usize]) -> DataMatrix {
    let mut data = Vec::with_capacity(indices.len() * 6);
    for &idx in indices {
        for d in 0..6 {
            data.push(correspondences.get(idx, d));
        }
    }
    DataMatrix::from_row_slice(indices.len(), 6, &data)
}
