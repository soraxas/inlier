use inlier::pointcloud::NormalEstimationStrategy;
use inlier::{
    ColoredIcpPipeline, ColoredIcpScale, ColoredIcpSettings, IcpConvergenceCriteria, PointCloud,
    PointCloudRegistrationPipeline, PointToPlaneKernel, RobustLoss,
};
use nalgebra::{DMatrix, Matrix4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_grid_cloud(rows: usize, cols: usize, spacing: f64) -> PointCloud {
    let mut data = Vec::with_capacity(rows * cols * 6);
    for i in 0..rows {
        for j in 0..cols {
            let x = i as f64 * spacing;
            let y = j as f64 * spacing;
            let z = 0.0;
            let r = (i as f64) / (rows.max(1) as f64);
            let g = (j as f64) / (cols.max(1) as f64);
            let b = 0.5;
            data.extend_from_slice(&[x, y, z, r, g, b]);
        }
    }
    let mat = DMatrix::from_row_slice(rows * cols, 6, &data);
    PointCloud::from_xyzrgb(&mat).expect("valid xyzrgb cloud")
}

fn make_sphere_cloud(count: usize) -> PointCloud {
    let mut rng = StdRng::seed_from_u64(123);
    let mut data = Vec::with_capacity(count * 6);
    for _ in 0..count {
        let z = rng.random_range(-1.0_f64..1.0);
        let t = rng.random_range(0.0_f64..(2.0 * std::f64::consts::PI));
        let r = (1.0_f64 - z * z).sqrt();
        let x = r * t.cos();
        let y = r * t.sin();
        let color = [(x + 1.0) * 0.5, (y + 1.0) * 0.5, (z + 1.0) * 0.5];
        data.extend_from_slice(&[x, y, z, color[0], color[1], color[2]]);
    }
    let mat = DMatrix::from_row_slice(count, 6, &data);
    PointCloud::from_xyzrgb(&mat).expect("valid xyzrgb cloud")
}

#[test]
fn voxel_downsample_reduces_points() {
    let cloud = make_grid_cloud(10, 10, 0.01);
    let down = cloud.voxel_downsample(0.05).expect("downsample");
    assert!(down.len() < cloud.len());
    assert!(down.colors.is_some());
}

#[test]
fn colored_icp_pipeline_runs_on_identity() {
    let source = make_grid_cloud(12, 12, 0.05);
    let target = make_grid_cloud(12, 12, 0.05);

    let scales = vec![ColoredIcpScale {
        voxel_radius: 0.1,
        max_iteration: 5,
    }];
    let criteria = IcpConvergenceCriteria {
        relative_fitness: 1e-6,
        relative_rmse: 1e-6,
        max_iteration: 5,
    };
    let settings = ColoredIcpSettings {
        distance_threshold: 0.2,
        normal_radius: 0.1,
        gradient_radius: 0.1,
        robust_loss: RobustLoss::Huber { delta: 0.02 },
        ..ColoredIcpSettings::default()
    };

    let mut pipeline = ColoredIcpPipeline::new()
        .with_scales(scales)
        .with_criteria(criteria)
        .with_settings(settings);

    let result = pipeline
        .run(&source, &target, Matrix4::identity())
        .expect("registration should succeed");

    assert!(result.fitness > 0.5);
    assert!(result.inlier_rmse < 1e-3);
}

#[test]
fn point_to_plane_pipeline_runs_without_color() {
    let source = {
        let cloud = make_grid_cloud(10, 10, 0.04);
        PointCloud::from_xyz(&cloud.points).expect("xyz cloud")
    };
    let target = PointCloud::from_xyz(&source.points).expect("xyz cloud");

    let scales = vec![ColoredIcpScale {
        voxel_radius: 0.08,
        max_iteration: 5,
    }];
    let settings = ColoredIcpSettings {
        distance_threshold: 0.15,
        normal_radius: 0.08,
        gradient_radius: 0.08,
        robust_loss: RobustLoss::None,
        ..ColoredIcpSettings::default()
    };

    let mut pipeline = ColoredIcpPipeline::new()
        .with_scales(scales)
        .with_settings(settings)
        .with_kernel(Box::new(PointToPlaneKernel));

    let result = pipeline
        .run(&source, &target, Matrix4::identity())
        .expect("registration should succeed");

    assert!(result.fitness > 0.5);
}

#[test]
fn fpfh_feature_shape_is_valid() {
    let cloud = make_grid_cloud(6, 6, 0.04);
    let mut cloud = cloud.clone();
    let normal_estimator = inlier::pointcloud::KnnNormalEstimator;
    cloud = normal_estimator
        .estimate(&cloud, 0.08, 30)
        .expect("normals");

    let features = inlier::compute_fpfh_features(
        &cloud,
        &inlier::FpfhSettings {
            radius: 0.12,
            max_nn: 50,
        },
    )
    .expect("fpfh");
    assert_eq!(features.nrows(), cloud.len());
    assert_eq!(features.ncols(), 33);
}

#[test]
fn global_to_local_pipeline_runs() {
    let source = make_sphere_cloud(500);
    let target = make_sphere_cloud(500);

    let mut pipeline = PointCloudRegistrationPipeline::new();
    pipeline.preprocess.voxel_size = 0.05;
    pipeline.preprocess.normal_radius = 0.1;
    pipeline.preprocess.fpfh_radius = 0.25;
    pipeline.global.distance_threshold = 0.08;
    pipeline.global.max_iterations = 800;
    pipeline.global.mutual_filter = false;
    pipeline.global.max_correspondences = 2000;

    let result = pipeline.run(&source, &target).expect("pipeline should run");
    assert!(result.global.fitness > 0.05);
    assert!(result.local.fitness > 0.1);
}
