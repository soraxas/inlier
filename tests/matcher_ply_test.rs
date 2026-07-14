//! Integration tests for KISS-Matcher with repo-local PLY fixtures.

#[cfg(feature = "io")]
use inlier::io::load_ply;
#[cfg(feature = "io")]
use inlier::matcher::{KISSMatcherConfig, kiss_matcher_full_pipeline};

#[test]
#[cfg(feature = "io")]
fn kiss_matcher_accepts_small_ply_fixture() {
    let src = load_ply("tests/data/sample_src.ply").expect("Failed to load source PLY");
    let dst = load_ply("tests/data/sample_dst.ply").expect("Failed to load destination PLY");

    assert_eq!(src.n_points(), 5);
    assert_eq!(dst.n_points(), 5);

    let config = KISSMatcherConfig {
        voxel_size: 0.05,
        normal_radius: 0.15,
        fpfh_radius: 0.3,
        the_linearity: 10.0,
        robin_noise_bound: 0.05,
        solver_noise_bound: 0.01,
        ratio_threshold: 0.9,
        ..Default::default()
    };

    if let Some(res) = kiss_matcher_full_pipeline(&src, &dst, &config) {
        assert!(res.scale.is_finite() && res.scale > 0.0);
        assert!(res.rotation.iter().all(|v| v.is_finite()));
        assert!(res.translation.iter().all(|v| v.is_finite()));
        assert!(res.n_correspondences_initial >= res.n_correspondences_final);
    }
}

#[test]
#[cfg(feature = "io")]
fn kiss_matcher_rejects_too_small_ply_clouds() {
    let src = load_ply("tests/data/sample_src.ply").expect("Failed to load source PLY");
    let dst = load_ply("tests/data/sample_dst.ply").expect("Failed to load destination PLY");
    let src_tiny = src.filter_points(&[true, true, false, false, false]);
    let dst_tiny = dst.filter_points(&[true, true, false, false, false]);

    assert!(kiss_matcher_full_pipeline(&src_tiny, &dst, &KISSMatcherConfig::default()).is_none());
    assert!(kiss_matcher_full_pipeline(&src, &dst_tiny, &KISSMatcherConfig::default()).is_none());
}

#[test]
#[cfg(not(feature = "io"))]
fn test_io_feature_required() {
    // Placeholder test when io feature is disabled
}
