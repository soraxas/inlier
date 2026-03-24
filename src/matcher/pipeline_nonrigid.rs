//! Non-rigid KISS-Matcher pipeline with RBF scale field estimation

use crate::estimators::rbf_scale_field::{NonRigidTransform, RBFScaleConfig, RBFScaleEstimator};
use crate::matcher::config::KISSMatcherConfig;
use crate::matcher::correspondence::FeatureMatcher;
use crate::matcher::features::{FasterPFH, FeaturePoint};
use crate::matcher::sipfh::{SIPFH, SIPFHConfig};
use crate::types::{DataMatrix, Point3};

/// Feature extraction method for non-rigid registration
#[derive(Clone, Debug)]
pub enum FeatureMethod {
    /// FasterPFH (fast but not scale-invariant)
    FasterPFH,
    /// SIPFH (scale-invariant, recommended for non-rigid)
    SIPFH(SIPFHConfig),
}

impl Default for FeatureMethod {
    fn default() -> Self {
        Self::SIPFH(SIPFHConfig::default())
    }
}

/// Configuration for non-rigid KISS-Matcher pipeline
#[derive(Clone, Debug, Default)]
pub struct NonRigidKISSConfig {
    /// Base KISS-Matcher parameters
    pub base: KISSMatcherConfig,
    /// RBF scale field configuration
    pub rbf: RBFScaleConfig,
    /// Feature extraction method
    pub feature_method: FeatureMethod,
    /// GNC max iterations
    pub gnc_max_iterations: usize,
    /// GNC final threshold multiplier (final = noise_bound * this)
    pub gnc_final_threshold_multiplier: f64,
}

/// Result from non-rigid KISS-Matcher pipeline
#[derive(Clone, Debug)]
pub struct NonRigidKISSResult {
    pub transform: NonRigidTransform,
    pub inlier_indices: Vec<usize>,
    pub n_correspondences_initial: usize,
    pub n_correspondences_after_robin: usize,
    pub n_correspondences_final: usize,
    pub mean_scale: f64,
    pub scale_std: f64,
    /// Source keypoints (downsampled cloud)
    pub source_keypoints: DataMatrix,
    /// Target keypoints (downsampled cloud)
    pub target_keypoints: DataMatrix,
    /// Initial correspondences (src_idx, tgt_idx)
    pub correspondences: Vec<(usize, usize)>,
}

/// Run non-rigid KISS-Matcher pipeline with spatially-varying scale
///
/// # Pipeline
/// 1. Voxel downsampling (if needed)
/// 2. Feature extraction (FasterPFH)
/// 3. Feature matching with ratio test
/// 4. RBF scale field estimation
/// 5. ROBIN pruning with per-point scales
/// 6. GNC refinement (optional)
pub fn nonrigid_kiss_matcher_pipeline(
    src_points: &DataMatrix,
    dst_points: &DataMatrix,
    config: &NonRigidKISSConfig,
) -> Option<NonRigidKISSResult> {
    use crate::preprocessing::voxel_downsample;

    if src_points.n_points() < 3 || dst_points.n_points() < 3 {
        return None;
    }

    println!("\n=== Non-Rigid KISS-Matcher Pipeline ===");
    println!(
        "Source: {} points, Target: {} points",
        src_points.n_points(),
        dst_points.n_points()
    );

    // Step 0: Downsample if clouds are too dense
    let (src_ds, dst_ds) = if src_points.n_points() > 50000 || dst_points.n_points() > 50000 {
        println!("\n[0/6] Downsampling dense point clouds...");
        let src_downsampled = voxel_downsample(src_points, config.base.voxel_size);
        let dst_downsampled = voxel_downsample(dst_points, config.base.voxel_size);
        println!(
            "  Downsampled to {} source, {} target points",
            src_downsampled.n_points(),
            dst_downsampled.n_points()
        );
        (src_downsampled, dst_downsampled)
    } else {
        (src_points.clone(), dst_points.clone())
    };

    // Step 1: Extract features (FasterPFH or SIPFH)
    let (src_features, dst_features) = match &config.feature_method {
        FeatureMethod::FasterPFH => {
            println!("\n[1/6] Extracting FasterPFH features...");
            let fpfh = FasterPFH::new(
                config.base.normal_radius,
                config.base.fpfh_radius,
                config.base.the_linearity,
                11,
            );

            let src_f = fpfh.compute_features(&src_ds);
            let dst_f = fpfh.compute_features(&dst_ds);

            println!(
                "  Extracted {} source features, {} target features",
                src_f.len(),
                dst_f.len()
            );

            (src_f, dst_f)
        }
        FeatureMethod::SIPFH(sipfh_config) => {
            println!("\n[1/6] Extracting SIPFH (scale-invariant) features...");
            let sipfh = SIPFH::new(sipfh_config.clone());

            let src_sipfh = sipfh.extract_features(&src_ds);
            let dst_sipfh = sipfh.extract_features(&dst_ds);

            println!(
                "  Extracted {} source keypoints, {} target keypoints",
                src_sipfh.len(),
                dst_sipfh.len()
            );

            // Convert SIPFH features to FeaturePoint for matching
            let src_f: Vec<FeaturePoint> = src_sipfh
                .iter()
                .map(|sf| FeaturePoint {
                    point: sf.feature.point,
                    normal: sf.feature.normal,
                    descriptor: sf.sipfh_descriptor.clone(), // Use SIPFH descriptor
                    is_valid: true,
                })
                .collect();

            let dst_f: Vec<FeaturePoint> = dst_sipfh
                .iter()
                .map(|sf| FeaturePoint {
                    point: sf.feature.point,
                    normal: sf.feature.normal,
                    descriptor: sf.sipfh_descriptor.clone(),
                    is_valid: true,
                })
                .collect();

            (src_f, dst_f)
        }
    };

    if src_features.is_empty() || dst_features.is_empty() {
        println!("  ERROR: No features extracted");
        return None;
    }

    // Step 2: Match features
    println!("\n[2/6] Matching features...");
    let matcher = FeatureMatcher::new(config.base.ratio_threshold);
    let correspondences = matcher.match_features(&src_features, &dst_features);

    println!("  Found {} initial correspondences", correspondences.len());

    if correspondences.len() < 3 {
        println!("  ERROR: Too few correspondences");
        return None;
    }

    let n_initial = correspondences.len();

    // Extract 3D point correspondences for RBF estimation
    let src_corr_points: Vec<Point3> = correspondences
        .iter()
        .map(|c| src_features[c.src_idx].point)
        .collect();

    let dst_corr_points: Vec<Point3> = correspondences
        .iter()
        .map(|c| dst_features[c.tgt_idx].point)
        .collect();

    // Step 3: Estimate RBF scale field
    println!("\n[3/6] Estimating RBF scale field...");
    let rbf_estimator = RBFScaleEstimator::new(config.rbf.clone());
    let mut transform = rbf_estimator.estimate(&src_corr_points, &dst_corr_points)?;

    let mean_scale = transform.mean_scale();
    let scale_std = compute_scale_std(&transform, &src_corr_points);

    println!("  Mean scale: {mean_scale:.6}");
    println!("  Scale std dev: {scale_std:.6}");
    println!(
        "  Control points: {}",
        transform.scale_field.control_points.len()
    );

    // Warn if scale variation is very large
    let scale_variation_pct = scale_std / mean_scale * 100.0;
    if scale_variation_pct > 30.0 {
        println!("\n  ⚠ Warning: Large scale variation ({scale_variation_pct:.1}%)!");
        println!("     This may indicate:");
        println!("     - Point clouds are rigid (no real deformation)");
        println!("     - RBF overfitting to outlier correspondences");
        println!("     - Consider using rigid KISS-Matcher instead");
    }

    // Step 4: ROBIN pruning with per-point scales
    println!("\n[4/6] ROBIN k-core pruning with per-point scales...");
    let inlier_indices = robin_prune_with_scale_field(
        &src_corr_points,
        &dst_corr_points,
        &transform,
        config.base.robin_noise_bound,
    );

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

    // Step 5: GNC refinement with graduated threshold
    println!("\n[5/6] GNC refinement (graduated non-convexity)...");

    // Extract inlier points from ROBIN
    let mut current_src: Vec<Point3> = inlier_indices.iter().map(|&i| src_corr_points[i]).collect();
    let mut current_dst: Vec<Point3> = inlier_indices.iter().map(|&i| dst_corr_points[i]).collect();

    // GNC parameters from config
    let max_iterations = config.gnc_max_iterations;
    let final_threshold = config.base.solver_noise_bound * config.gnc_final_threshold_multiplier;

    // Compute initial residuals to set starting threshold
    let initial_residuals: Vec<f64> = current_src
        .iter()
        .zip(current_dst.iter())
        .map(|(s, d)| transform.residual(s, d))
        .collect();

    // Start with threshold at 90th percentile of residuals (accept 90% initially)
    let mut sorted_init = initial_residuals.clone();
    sorted_init.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let initial_threshold = sorted_init[(sorted_init.len() * 9 / 10).min(sorted_init.len() - 1)]
        .max(final_threshold * 2.0); // At least 2× final threshold

    println!("  Initial threshold: {initial_threshold:.4} (90th percentile)");
    println!("  Final threshold: {final_threshold:.4}");
    println!("  Max iterations: {max_iterations}");

    let mut best_transform = transform.clone();
    let mut best_inlier_count = 0;
    let mut best_current_src = current_src.clone();
    let mut best_current_dst = current_dst.clone();
    let mut stopped_early = false;

    for iteration in 0..max_iterations {
        // Graduated threshold: exponential decay from initial to final
        let t = iteration as f64 / (max_iterations - 1) as f64;
        let current_threshold = initial_threshold * (final_threshold / initial_threshold).powf(t);

        // Compute residuals
        let residuals: Vec<f64> = current_src
            .iter()
            .zip(current_dst.iter())
            .map(|(s, d)| transform.residual(s, d))
            .collect();

        // Count inliers at current threshold
        let inlier_mask: Vec<bool> = residuals.iter().map(|&r| r <= current_threshold).collect();
        let inlier_count = inlier_mask.iter().filter(|&&b| b).count();

        let max_residual = residuals.iter().cloned().fold(0.0f64, f64::max);
        let median_residual = {
            let mut sorted = residuals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        println!(
            "  Iter {:2}: threshold={:.4}, inliers={:3}/{:3} ({:5.1}%), median_res={:.4}, max_res={:.4}",
            iteration,
            current_threshold,
            inlier_count,
            current_src.len(),
            100.0 * inlier_count as f64 / current_src.len() as f64,
            median_residual,
            max_residual
        );

        // Track best model
        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_transform = transform.clone();
            best_current_src = current_src.clone();
            best_current_dst = current_dst.clone();
        }

        // Early stopping criteria

        // 1. Stop if we're at final threshold AND have high inlier ratio
        let at_final_threshold = (current_threshold - final_threshold).abs() < 0.001;
        if at_final_threshold && inlier_count >= current_src.len() * 95 / 100 {
            println!("  → Converged at final threshold (95% inliers)");
            break;
        }

        // 2. Stop if model is good: median residual reasonable and enough inliers
        let min_inliers_for_early_stop = (n_after_robin / 10).max(50); // At least 10% of ROBIN inliers or 50
        if iteration >= 3 && inlier_count >= min_inliers_for_early_stop {
            // Check if median residual is reasonable (within current threshold range)
            // We want median < current_threshold (most points fit current tolerance)
            if median_residual < current_threshold * 1.2 {
                println!(
                    "  → Early stop: {inlier_count} inliers with acceptable residual {median_residual:.4} (threshold {current_threshold:.4})"
                );
                stopped_early = true;
                // Keep current inliers - they're good at this threshold!
                break;
            }
        }

        // 3. Stop if inlier count drops too much (>70% loss from previous iteration)
        if iteration > 0 && inlier_count < current_src.len() * 3 / 10 {
            println!(
                "  → Stopping: too many inliers lost ({} → {})",
                current_src.len(),
                inlier_count
            );
            // Use best model so far
            transform = best_transform.clone();
            current_src = best_current_src.clone();
            current_dst = best_current_dst.clone();
            stopped_early = true;
            break;
        }

        // 4. Stop if too few inliers
        if inlier_count < 3 {
            println!("  → Failed (too few inliers)");
            break;
        }

        // Filter to inliers for next iteration
        let inlier_src: Vec<Point3> = current_src
            .iter()
            .zip(inlier_mask.iter())
            .filter_map(|(p, &keep)| if keep { Some(*p) } else { None })
            .collect();
        let inlier_dst: Vec<Point3> = current_dst
            .iter()
            .zip(inlier_mask.iter())
            .filter_map(|(p, &keep)| if keep { Some(*p) } else { None })
            .collect();

        if inlier_src.len() < 3 {
            break;
        }

        // Check if we have enough points vs control points for stable estimation
        let control_to_point_ratio = inlier_src.len() as f64 / config.rbf.num_control_points as f64;
        if control_to_point_ratio < 3.0 {
            println!(
                "  → Stopping: too few points ({}) for {} control points (ratio {:.1}:1)",
                inlier_src.len(),
                config.rbf.num_control_points,
                control_to_point_ratio
            );
            break;
        }

        // Re-estimate with current inliers
        if let Some(refined) = rbf_estimator.estimate(&inlier_src, &inlier_dst) {
            // Check if re-estimation improves the model
            let old_residuals: Vec<f64> = inlier_src
                .iter()
                .zip(inlier_dst.iter())
                .map(|(s, d)| transform.residual(s, d))
                .collect();
            let new_residuals: Vec<f64> = inlier_src
                .iter()
                .zip(inlier_dst.iter())
                .map(|(s, d)| refined.residual(s, d))
                .collect();

            let old_median = {
                let mut sorted = old_residuals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            };
            let new_median = {
                let mut sorted = new_residuals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            };

            // Only accept if new model is better
            if new_median < old_median * 1.5 {
                // Allow up to 50% worse (some degradation OK)
                current_src = inlier_src;
                current_dst = inlier_dst;
                transform = refined;
            } else {
                println!(
                    "  → Re-estimation degraded model (median {old_median:.4} → {new_median:.4}), keeping previous"
                );
                break;
            }
        } else {
            println!("  → Re-estimation failed");
            break;
        }
    }

    // Use best model found (or current if we stopped early)
    if !stopped_early {
        transform = best_transform;
        current_src = best_current_src;
        current_dst = best_current_dst;
    }

    // When we stop early, current_src/current_dst ARE the final inliers
    // No need to re-validate with strict threshold
    let final_inliers: Vec<usize> = if stopped_early {
        // Use all current points as inliers
        (0..current_src.len()).collect()
    } else {
        // Compute final inliers at strict threshold
        compute_final_inliers(&current_src, &current_dst, &transform, final_threshold)
    };

    println!(
        "  Final inliers: {} ({:.1}%)",
        final_inliers.len(),
        100.0 * final_inliers.len() as f64 / n_after_robin as f64
    );

    // Check if we have enough final inliers
    if final_inliers.len() < 3 {
        println!("  ERROR: Too few final inliers after refinement");
        println!(
            "\nNote: Large scale variation ({:.1}% std) may indicate:",
            scale_std / mean_scale * 100.0
        );
        println!("  - Point clouds are actually rigid (no real deformation)");
        println!("  - RBF overfitting to outliers");
        println!("  - Need tighter ROBIN noise bound or more correspondences");
        return None;
    }

    println!("\n=== Results ===");
    println!("Mean scale: {:.6}", transform.mean_scale());
    println!(
        "Scale std dev: {:.6}",
        compute_scale_std(&transform, &current_src)
    );
    println!("Rotation:\n{}", transform.rotation);
    println!("Translation: {:?}", transform.translation);
    println!(
        "Total inliers: {}/{} ({:.1}%)",
        final_inliers.len(),
        n_initial,
        100.0 * final_inliers.len() as f64 / n_initial as f64
    );

    let n_final = final_inliers.len();

    // Build keypoint DataMatrix from actual feature points
    // (correspondences index into these, not src_ds/dst_ds!)
    // DataMatrix::from_row_slice expects ROW-MAJOR data: [x0,y0,z0, x1,y1,z1, ...]
    let src_keypoints = {
        let n = src_features.len();
        let mut points = Vec::with_capacity(n * 3);

        // Row-major order: x, y, z for each point
        for feat in &src_features {
            points.push(feat.point.x);
            points.push(feat.point.y);
            points.push(feat.point.z);
        }

        DataMatrix::from_row_slice(n, 3, &points)
    };

    let dst_keypoints = {
        let n = dst_features.len();
        let mut points = Vec::with_capacity(n * 3);

        // Row-major order: x, y, z for each point
        for feat in &dst_features {
            points.push(feat.point.x);
            points.push(feat.point.y);
            points.push(feat.point.z);
        }

        DataMatrix::from_row_slice(n, 3, &points)
    };

    // Convert correspondences to (src_idx, tgt_idx) format
    let corr_pairs: Vec<(usize, usize)> = correspondences
        .iter()
        .map(|c| (c.src_idx, c.tgt_idx))
        .collect();

    Some(NonRigidKISSResult {
        transform,
        inlier_indices: final_inliers,
        n_correspondences_initial: n_initial,
        n_correspondences_after_robin: n_after_robin,
        n_correspondences_final: n_final,
        mean_scale,
        scale_std,
        source_keypoints: src_keypoints,
        target_keypoints: dst_keypoints,
        correspondences: corr_pairs,
    })
}

/// ROBIN pruning with per-point scales from scale field
fn robin_prune_with_scale_field(
    src_points: &[Point3],
    dst_points: &[Point3],
    transform: &NonRigidTransform,
    noise_bound: f64,
) -> Vec<usize> {
    let n = src_points.len();

    // Build compatibility graph
    let mut compatible = vec![vec![false; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            // Compute TIMs with local scales
            let si = transform.scale_field.eval(&src_points[i]);
            let sj = transform.scale_field.eval(&src_points[j]);

            let src_tim = (src_points[j] - src_points[i]).norm();
            let dst_tim = (dst_points[j] - dst_points[i]).norm();

            // Use average scale for this pair
            let avg_scale = (si + sj) / 2.0;
            let scaled_src_tim = avg_scale * src_tim;

            // Check if compatible
            let residual = (scaled_src_tim - dst_tim).abs();
            if residual <= noise_bound {
                compatible[i][j] = true;
                compatible[j][i] = true;
            }
        }
    }

    // K-core decomposition (greedy)
    let mut active: Vec<bool> = vec![true; n];
    let mut changed = true;

    while changed {
        changed = false;
        for i in 0..n {
            if !active[i] {
                continue;
            }

            // Count active neighbors
            let degree = (0..n)
                .filter(|&j| i != j && active[j] && compatible[i][j])
                .count();

            // Remove if degree too low
            if degree < 2 {
                active[i] = false;
                changed = true;
            }
        }
    }

    (0..n).filter(|&i| active[i]).collect()
}

/// Compute standard deviation of scales
fn compute_scale_std(transform: &NonRigidTransform, points: &[Point3]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let scales: Vec<f64> = points
        .iter()
        .map(|p| transform.scale_field.eval(p))
        .collect();
    let mean = scales.iter().sum::<f64>() / scales.len() as f64;
    let variance = scales.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scales.len() as f64;
    variance.sqrt()
}

/// Compute final inliers based on residual threshold
fn compute_final_inliers(
    src_points: &[Point3],
    dst_points: &[Point3],
    transform: &NonRigidTransform,
    threshold: f64,
) -> Vec<usize> {
    src_points
        .iter()
        .zip(dst_points.iter())
        .enumerate()
        .filter_map(|(i, (src, dst))| {
            let residual = transform.residual(src, dst);
            if residual <= threshold { Some(i) } else { None }
        })
        .collect()
}
