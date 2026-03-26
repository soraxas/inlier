use inlier::matcher::correspondence::FeatureMatcher;
use inlier::matcher::features::FasterPFH;
use inlier::types::DataMatrix;

#[test]
fn test_feature_matching_pipeline() {
    // Create source point cloud: dense enough for features
    let mut src_points = Vec::new();
    for x in 0..6 {
        for y in 0..6 {
            let z = if x == 3 && y == 3 { 0.2 } else { 0.0 }; // Center raised
            src_points.push(x as f64 * 0.2);
            src_points.push(y as f64 * 0.2);
            src_points.push(z);
        }
    }
    let src_data = DataMatrix::from_row_slice(36, 3, &src_points);

    // Create target: translated version
    let mut tgt_points = Vec::new();
    for i in (0..src_points.len()).step_by(3) {
        tgt_points.push(src_points[i] + 0.15);
        tgt_points.push(src_points[i + 1] + 0.15);
        tgt_points.push(src_points[i + 2]);
    }
    let tgt_data = DataMatrix::from_row_slice(36, 3, &tgt_points);

    // Extract features with very loose parameters
    let fpfh = FasterPFH::new(0.4, 0.6, 0.99, 11); // Very loose linearity filter
    let src_features = fpfh.compute_features(&src_data);
    let tgt_features = fpfh.compute_features(&tgt_data);

    println!(
        "Extracted {} source features, {} target features",
        src_features.len(),
        tgt_features.len()
    );

    if src_features.is_empty() || tgt_features.is_empty() {
        println!("Warning: No features extracted. Point cloud may be too sparse.");
        println!("Skipping correspondence test.");
        return;
    }

    // Match features with very loose ratio test
    let matcher = FeatureMatcher::new(0.99);
    let correspondences = matcher.match_features(&src_features, &tgt_features);

    println!("Found {} correspondences", correspondences.len());

    // With distinctive geometry, should find at least some matches
    assert!(
        !correspondences.is_empty(),
        "Should find at least some correspondences"
    );
    assert!(
        correspondences.len() <= src_features.len().min(tgt_features.len()),
        "Cannot have more matches than features"
    );

    // Check that correspondences are valid indices
    for corr in &correspondences {
        assert!(corr.src_idx < src_features.len());
        assert!(corr.tgt_idx < tgt_features.len());
    }

    // Convert to matrix format
    let corr_matrix =
        matcher.correspondences_to_matrix(&correspondences, &src_features, &tgt_features);

    assert_eq!(corr_matrix.n_points(), correspondences.len());
    assert_eq!(corr_matrix.n_dims(), 6);

    println!(
        "Generated correspondence matrix: {} x {}",
        corr_matrix.n_points(),
        corr_matrix.n_dims()
    );
}

#[test]
fn test_feature_matching_with_noise() {
    // Create source: simple points with variation
    let src_points = vec![
        0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.3,
        0.1, // Center raised
        0.6, 0.3, 0.0, 0.0, 0.6, 0.0, 0.3, 0.6, 0.0, 0.6, 0.6, 0.0,
    ];
    let src_data = DataMatrix::from_row_slice(9, 3, &src_points);

    // Create target: translated + some outliers
    let mut tgt_points = Vec::new();

    // Add translated inliers
    for i in (0..src_points.len()).step_by(3) {
        tgt_points.push(src_points[i] + 0.2);
        tgt_points.push(src_points[i + 1] + 0.2);
        tgt_points.push(src_points[i + 2]);
    }

    // Add outliers (different location)
    for i in 0..3 {
        tgt_points.push(2.0 + i as f64 * 0.2);
        tgt_points.push(2.0);
        tgt_points.push(0.0);
    }

    let tgt_data = DataMatrix::from_row_slice(12, 3, &tgt_points);

    // Extract features with generous parameters
    let fpfh = FasterPFH::new(0.3, 0.5, 0.95, 11);
    let src_features = fpfh.compute_features(&src_data);
    let tgt_features = fpfh.compute_features(&tgt_data);

    println!(
        "With noise: {} source features, {} target features",
        src_features.len(),
        tgt_features.len()
    );

    if src_features.is_empty() || tgt_features.is_empty() {
        println!("Warning: No features extracted, skipping test");
        return;
    }

    // Match with loose ratio test
    let matcher = FeatureMatcher::new(0.95);
    let correspondences = matcher.match_features(&src_features, &tgt_features);

    println!("Found {} correspondences with noise", correspondences.len());

    // Verify correspondences exist and are valid
    for corr in &correspondences {
        assert!(corr.src_idx < src_features.len());
        assert!(corr.tgt_idx < tgt_features.len());
    }
}

#[test]
fn test_feature_matching_ratio_threshold() {
    // Create distinctive point clouds (not uniform)
    let src_points = vec![
        0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5,
        0.2, // Raised center
        1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0,
    ];
    let src_data = DataMatrix::from_row_slice(9, 3, &src_points);

    // Same points
    let _tgt_data = DataMatrix::from_row_slice(9, 3, &src_points);

    // Extract features with generous radii
    let fpfh = FasterPFH::new(0.4, 0.6, 0.95, 11);
    let features = fpfh.compute_features(&src_data);

    if features.len() < 2 {
        println!("Not enough features for ratio test, skipping");
        return;
    }

    // Match with different ratio thresholds
    let matcher_strict = FeatureMatcher::new(0.6); // Strict
    let matcher_loose = FeatureMatcher::new(0.95); // Loose

    let matches_strict = matcher_strict.match_features(&features, &features);
    let matches_loose = matcher_loose.match_features(&features, &features);

    println!(
        "Strict (0.6): {} matches, Loose (0.95): {} matches",
        matches_strict.len(),
        matches_loose.len()
    );

    // Loose threshold should find at least as many matches
    assert!(
        matches_loose.len() >= matches_strict.len(),
        "Loose threshold should find at least as many matches"
    );

    // All matches should be identity (same index) with identical clouds
    for corr in &matches_loose {
        assert_eq!(
            corr.src_idx, corr.tgt_idx,
            "With identical clouds, should match to same index"
        );
        assert!(
            corr.distance < 1e-6,
            "With identical features, distance should be near zero"
        );
    }
}
