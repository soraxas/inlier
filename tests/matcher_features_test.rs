use inlier::matcher::features::FasterPFH;
use inlier::types::DataMatrix;

#[test]
fn test_fpfh_on_simple_point_cloud() {
    // Create a simple point cloud - a 5x5 grid on the XY plane
    let mut points = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            points.push(x as f64 * 0.1);
            points.push(y as f64 * 0.1);
            points.push(0.0);
        }
    }

    let data = DataMatrix::from_row_slice(25, 3, &points);

    // Extract features with generous parameters
    let fpfh = FasterPFH::new(0.15, 0.25, 0.8, 11);
    let features = fpfh.compute_features(&data);

    // Should extract most points (grid is planar, so linearity filter applies)
    println!(
        "Extracted {} features from {} points",
        features.len(),
        data.n_points()
    );
    assert!(
        features.len() >= 15,
        "Should extract at least 15 features from planar grid"
    );

    // Check that features have valid normals
    for feat in &features {
        let norm_length = feat.normal.norm();
        assert!(
            (norm_length - 1.0).abs() < 0.01,
            "Normal should be unit length"
        );
    }

    // Check that FPFH descriptors are computed
    for feat in &features {
        assert_eq!(feat.descriptor.len(), 33, "FPFH should have 33 bins");
        let sum: f64 = feat.descriptor.iter().sum();
        assert!(sum > 0.0, "FPFH descriptor should not be all zeros");
    }
}

#[test]
fn test_fpfh_normal_estimation() {
    // Create points on a tilted plane: z = 0.5*x + 0.3*y
    let mut points = Vec::new();
    for x in 0..10 {
        for y in 0..10 {
            let x_val = x as f64 * 0.1;
            let y_val = y as f64 * 0.1;
            let z_val = 0.5 * x_val + 0.3 * y_val;
            points.push(x_val);
            points.push(y_val);
            points.push(z_val);
        }
    }

    let data = DataMatrix::from_row_slice(100, 3, &points);

    // Extract features
    let fpfh = FasterPFH::new(0.2, 0.35, 0.9, 11);
    let features = fpfh.compute_features(&data);

    println!("Extracted {} features from tilted plane", features.len());

    // All normals should point roughly in the same direction for a planar surface
    if features.len() >= 2 {
        let first_normal = &features[0].normal;

        for feat in &features[1..] {
            // Compute angle between normals (dot product)
            // Normals may point opposite directions, so use abs
            let dot = first_normal.dot(&feat.normal).abs();
            let dot_clamped = dot.clamp(-1.0, 1.0);
            let angle = dot_clamped.acos().to_degrees();

            // Should be within 30 degrees for planar surface
            assert!(
                angle < 30.0,
                "Normals on plane should be consistent, got {angle} degrees"
            );
        }
    }
}

#[test]
fn test_fpfh_with_different_voxel_sizes() {
    // Create a dense point cloud
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            points.push(x as f64 * 0.05);
            points.push(y as f64 * 0.05);
            points.push((x + y) as f64 * 0.02);
        }
    }

    let data = DataMatrix::from_row_slice(400, 3, &points);

    // Small radius - more features
    let fpfh_small = FasterPFH::new(0.1, 0.2, 0.8, 11);
    let features_small = fpfh_small.compute_features(&data);

    // Large radius - fewer features (more get filtered)
    let fpfh_large = FasterPFH::new(0.3, 0.5, 0.8, 11);
    let features_large = fpfh_large.compute_features(&data);

    println!("Small radius: {} features", features_small.len());
    println!("Large radius: {} features", features_large.len());

    // Both should extract reasonable number of features
    assert!(
        !features_small.is_empty(),
        "Should extract features with small radius"
    );
    assert!(
        !features_large.is_empty(),
        "Should extract features with large radius"
    );
}

#[test]
fn test_fpfh_sparse_point_cloud() {
    // Very sparse point cloud - just 5 points
    let points = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5,
    ];

    let data = DataMatrix::from_row_slice(5, 3, &points);

    // With large radius, should find neighbors
    let fpfh = FasterPFH::new(0.8, 1.5, 0.95, 11);
    let features = fpfh.compute_features(&data);

    println!("Extracted {} features from 5 sparse points", features.len());

    // With very sparse data, we might not extract many features
    // Just check that it doesn't crash
    assert!(
        features.len() <= 5,
        "Cannot extract more features than points"
    );
}
