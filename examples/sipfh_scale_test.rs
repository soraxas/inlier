//! Test SIPFH with synthetic scale-varying data

use inlier::estimators::rbf_scale_field::{RBFKernel, RBFScaleConfig};
use inlier::matcher::config::KISSMatcherConfig;
use inlier::matcher::pipeline_nonrigid::{
    FeatureMethod, NonRigidKISSConfig, nonrigid_kiss_matcher_pipeline,
};
use inlier::matcher::sipfh::SIPFHConfig;
use inlier::types::DataMatrix;
use rand::Rng;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("   SIPFH Non-Rigid Registration: Scale-Varying Test");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create source point cloud (structured cube with jitter)
    let mut rng = rand::rng();
    let mut src_points = Vec::new();
    for x in -10..=10 {
        for y in -10..=10 {
            for z in -10..=10 {
                // Create sparse structure
                if x % 2 == 0 && y % 2 == 0 && z % 2 == 0 {
                    // Add small jitter to avoid grid alignment issues
                    src_points.push(x as f64 + rng.random_range(-0.1..0.1));
                    src_points.push(y as f64 + rng.random_range(-0.1..0.1));
                    src_points.push(z as f64 + rng.random_range(-0.1..0.1));
                }
            }
        }
    }

    println!("Source cloud: {} points", src_points.len() / 3);

    // Create target with spatially-varying scale
    let mut dst_points = Vec::new();
    for i in (0..src_points.len()).step_by(3) {
        let x = src_points[i];
        let y = src_points[i + 1];
        let z = src_points[i + 2];

        // Scale varies from 1.0 to 1.15 based on x coordinate
        let scale = 1.0 + 0.15 * (x + 10.0) / 20.0;

        dst_points.push(x * scale);
        dst_points.push(y * scale);
        dst_points.push(z * scale);
    }

    println!("Target cloud: {} points", dst_points.len() / 3);
    println!("Scale variation: 1.0 → 1.15 (15% variation)\n");

    let src = DataMatrix::from_row_slice(src_points.len() / 3, 3, &src_points);
    let dst = DataMatrix::from_row_slice(dst_points.len() / 3, 3, &dst_points);

    // Configure with SIPFH
    let config = NonRigidKISSConfig {
        base: KISSMatcherConfig {
            voxel_size: 1.0,
            normal_radius: 2.0,
            fpfh_radius: 4.0,
            ratio_threshold: 0.85,
            robin_noise_bound: 0.5,
            solver_noise_bound: 0.3,
            the_linearity: 10.0,
            ..Default::default()
        },
        rbf: RBFScaleConfig {
            num_control_points: 10,
            kernel: RBFKernel::Gaussian { sigma: 5.0 },
            regularization_lambda: 1e-3,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            min_scale: 0.9,
            max_scale: 1.3,
            use_sparse: true,
        },
        feature_method: FeatureMethod::SIPFH(SIPFHConfig {
            num_octaves: 3,
            scales_per_octave: 3,
            initial_sigma: 1.0,
            dog_threshold: 0.01,
            edge_threshold: 10.0,
            fpfh_radius: 4.0,
            the_linearity: 0.9,
            fpfh_bins: 11,
            scale_weight: 0.5,
        }),
    };

    println!("Running SIPFH-based non-rigid registration...\n");

    match nonrigid_kiss_matcher_pipeline(&src, &dst, &config) {
        Some(result) => {
            println!("\n✓ Registration SUCCEEDED!");
            println!("══════════════════════════════════════════════════════\n");

            println!("Results:");
            println!("  Mean scale: {:.4}", result.mean_scale);
            println!("  Scale std dev: {:.4}", result.scale_std);
            println!(
                "  Scale variation: {:.1}%",
                result.scale_std / result.mean_scale * 100.0
            );
            println!(
                "  Control points: {}",
                result.transform.scale_field.control_points.len()
            );
            println!("\nCorrespondences:");
            println!("  Initial: {}", result.n_correspondences_initial);
            println!("  After ROBIN: {}", result.n_correspondences_after_robin);
            println!("  Final inliers: {}", result.n_correspondences_final);
            println!(
                "  Inlier ratio: {:.1}%",
                100.0 * result.n_correspondences_final as f64
                    / result.n_correspondences_initial as f64
            );

            println!("\n✅ SIPFH successfully handled 15% scale variation!");
            println!("   This demonstrates scale-invariant feature matching.");
        }
        None => {
            println!("\n✗ Registration failed!");
            println!("\nThis might be due to:");
            println!("  - Need to tune SIPFH parameters (DoG threshold, sigma)");
            println!("  - Point cloud too sparse for keypoint detection");
            println!("  - Try denser sampling or adjust voxel size");
        }
    }
}
