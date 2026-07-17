//! Deterministic malformed-input benchmarks for public API rejection paths.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use inlier::{
    MetasacSettings, estimate_absolute_pose, estimate_essential_matrix,
    estimate_fundamental_matrix, estimate_homography, estimate_line, estimate_plane,
    estimate_rigid_transform,
    settings::{LocalOptimizationType, SamplerType, ScoringType},
    types::DataMatrix,
};
use std::time::Duration;

const POINTS: usize = 64;

fn rejection_settings() -> MetasacSettings {
    MetasacSettings {
        min_iterations: 64,
        max_iterations: 64,
        max_sampling_attempts: 1,
        rng_seed: Some(0xD3E3_0001),
        scoring: ScoringType::Msac,
        sampler: SamplerType::Uniform,
        local_optimization: LocalOptimizationType::None,
        final_optimization: LocalOptimizationType::None,
        ..Default::default()
    }
}

fn collinear_correspondences() -> (DataMatrix, DataMatrix) {
    let mut source = DataMatrix::zeros(POINTS, 2);
    let mut target = DataMatrix::zeros(POINTS, 2);
    for index in 0..POINTS {
        let x = index as f64;
        source.set(index, 0, x);
        source.set(index, 1, 0.0);
        target.set(index, 0, x + 10.0);
        target.set(index, 1, 0.0);
    }
    (source, target)
}

fn repeated_points(dimensions: usize) -> DataMatrix {
    DataMatrix::zeros(POINTS, dimensions)
}

fn benchmark_rejection_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("rejection_paths");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let collapsed_line = repeated_points(2);
    group.bench_function("line_collapsed", |bench| {
        bench.iter(|| {
            assert!(
                estimate_line(black_box(&collapsed_line), 0.1, Some(rejection_settings())).is_err()
            );
        });
    });

    let collapsed_plane = repeated_points(3);
    group.bench_function("plane_collapsed", |bench| {
        bench.iter(|| {
            assert!(
                estimate_plane(black_box(&collapsed_plane), 0.1, Some(rejection_settings()))
                    .is_err()
            );
        });
    });

    let (homography_source, homography_target) = collinear_correspondences();
    group.bench_function("homography_collinear", |bench| {
        bench.iter(|| {
            assert!(
                estimate_homography(
                    black_box(&homography_source),
                    black_box(&homography_target),
                    1.0,
                    Some(rejection_settings()),
                )
                .is_err()
            );
        });
    });

    let collapsed_image = repeated_points(2);
    group.bench_function("fundamental_collapsed", |bench| {
        bench.iter(|| {
            assert!(
                estimate_fundamental_matrix(
                    black_box(&collapsed_image),
                    black_box(&collapsed_image),
                    0.01,
                    Some(rejection_settings()),
                )
                .is_err()
            );
        });
    });
    group.bench_function("essential_collapsed", |bench| {
        bench.iter(|| {
            assert!(
                estimate_essential_matrix(
                    black_box(&collapsed_image),
                    black_box(&collapsed_image),
                    0.01,
                    Some(rejection_settings()),
                )
                .is_err()
            );
        });
    });

    let collapsed_world = repeated_points(3);
    group.bench_function("absolute_pose_collapsed", |bench| {
        bench.iter(|| {
            assert!(
                estimate_absolute_pose(
                    black_box(&collapsed_world),
                    black_box(&collapsed_image),
                    0.01,
                    Some(rejection_settings()),
                )
                .is_err()
            );
        });
    });
    group.bench_function("rigid_transform_collapsed", |bench| {
        bench.iter(|| {
            assert!(
                estimate_rigid_transform(
                    black_box(&collapsed_world),
                    black_box(&collapsed_world),
                    0.01,
                    Some(rejection_settings()),
                )
                .is_err()
            );
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_rejection_paths);
criterion_main!(benches);
