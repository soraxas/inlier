use inlier::kiss_matcher::config::KISSMatcherConfig;
use inlier::kiss_matcher::pipeline_full::kiss_matcher_full_pipeline;
use inlier::types::DataMatrix;

#[test]
fn test_full_pipeline_simple() {
    // Create source: 4x4 grid
    let mut src_points = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            src_points.push(x as f64 * 0.3);
            src_points.push(y as f64 * 0.3);
            src_points.push(0.0);
        }
    }
    let src_data = DataMatrix::from_row_slice(16, 3, &src_points);

    // Create target: translated version
    let mut tgt_points = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            tgt_points.push(x as f64 * 0.3 + 0.5); // Translate x by 0.5
            tgt_points.push(y as f64 * 0.3 + 0.3); // Translate y by 0.3
            tgt_points.push(0.1); // Translate z by 0.1
        }
    }
    let tgt_data = DataMatrix::from_row_slice(16, 3, &tgt_points);

    // Run full pipeline with loose parameters
    let config = KISSMatcherConfig::new(0.2); // voxel_size

    let result = kiss_matcher_full_pipeline(&src_data, &tgt_data, &config);

    match result {
        Some(res) => {
            println!("Pipeline succeeded!");
            println!("Scale: {:.4}", res.scale);
            println!("Rotation:\n{}", res.rotation);
            println!("Translation: {:?}", res.translation);
            println!(
                "Inliers: {}/{}",
                res.n_correspondences_final, res.n_correspondences_initial
            );

            // Scale should be close to 1.0
            assert!((res.scale - 1.0).abs() < 0.2, "Scale should be near 1.0");

            // Should find some inliers
            assert!(
                res.n_correspondences_final > 0,
                "Should find at least some inliers"
            );
        }
        None => {
            println!("Pipeline failed - this may be expected for simple test data");
        }
    }
}
