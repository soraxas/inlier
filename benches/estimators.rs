//! Deterministic end-to-end benchmarks for the public robust-estimation APIs.
//!
//! The scenes are intentionally generated in-memory. Large, real-world benchmark inputs belong
//! in the `inlier-data` submodule; these compact synthetic scenes make performance regressions
//! reproducible on every developer machine.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use inlier::{
    MetasacSettings, estimate_absolute_pose, estimate_essential_matrix,
    estimate_fundamental_matrix, estimate_homography, estimate_line, estimate_plane,
    estimate_rigid_transform, types::DataMatrix,
};
use nalgebra::{Matrix3, Rotation3, Vector3};
use std::time::Duration;

const SEED: u64 = 0x5EED_CAFE_D00D_BAAD;

#[derive(Clone, Copy)]
enum Scene {
    Clean,
    Noisy,
    Outliers,
}

impl Scene {
    const ALL: [Self; 3] = [Self::Clean, Self::Noisy, Self::Outliers];

    fn name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Noisy => "noisy",
            Self::Outliers => "noisy_25pct_outliers",
        }
    }

    fn noise(self) -> f64 {
        match self {
            Self::Clean => 0.0,
            Self::Noisy | Self::Outliers => 0.0001,
        }
    }

    fn is_outlier(self, index: usize) -> bool {
        matches!(self, Self::Outliers) && index.is_multiple_of(4)
    }
}

struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }

    fn signed_noise(&mut self, magnitude: f64) -> f64 {
        self.range(-magnitude, magnitude)
    }
}

fn settings(seed: u64) -> MetasacSettings {
    MetasacSettings {
        min_iterations: 400,
        max_iterations: 400,
        max_sampling_attempts: 100,
        rng_seed: Some(seed),
        ..Default::default()
    }
}

fn image_correspondences(scene: Scene, count: usize) -> (DataMatrix, DataMatrix) {
    let rotation = Rotation3::from_euler_angles(0.03, -0.12, 0.02);
    let translation = Vector3::new(0.35, -0.08, 0.12);
    let mut rng = DeterministicRng::new(SEED);
    let mut first = DataMatrix::zeros(count, 2);
    let mut second = DataMatrix::zeros(count, 2);

    for index in 0..count {
        let point = Vector3::new(
            rng.range(-1.5, 1.5),
            rng.range(-1.0, 1.0),
            rng.range(4.0, 8.0),
        );
        let transformed = rotation * point + translation;
        first.set(
            index,
            0,
            point.x / point.z + rng.signed_noise(scene.noise()),
        );
        first.set(
            index,
            1,
            point.y / point.z + rng.signed_noise(scene.noise()),
        );
        second.set(
            index,
            0,
            if scene.is_outlier(index) {
                rng.range(-1.0, 1.0)
            } else {
                transformed.x / transformed.z + rng.signed_noise(scene.noise())
            },
        );
        second.set(
            index,
            1,
            if scene.is_outlier(index) {
                rng.range(-1.0, 1.0)
            } else {
                transformed.y / transformed.z + rng.signed_noise(scene.noise())
            },
        );
    }

    (first, second)
}

fn homography_correspondences(scene: Scene, count: usize) -> (DataMatrix, DataMatrix) {
    let h = Matrix3::new(1.08, -0.06, 12.0, 0.04, 0.94, -8.0, 0.0008, -0.0005, 1.0);
    let mut rng = DeterministicRng::new(SEED ^ 0x1234);
    let mut first = DataMatrix::zeros(count, 2);
    let mut second = DataMatrix::zeros(count, 2);

    for index in 0..count {
        let x = rng.range(-100.0, 100.0);
        let y = rng.range(-80.0, 80.0);
        let mapped = h * Vector3::new(x, y, 1.0);
        first.set(index, 0, x);
        first.set(index, 1, y);
        second.set(
            index,
            0,
            if scene.is_outlier(index) {
                rng.range(-150.0, 150.0)
            } else {
                mapped.x / mapped.z + rng.signed_noise(scene.noise() * 100.0)
            },
        );
        second.set(
            index,
            1,
            if scene.is_outlier(index) {
                rng.range(-150.0, 150.0)
            } else {
                mapped.y / mapped.z + rng.signed_noise(scene.noise() * 100.0)
            },
        );
    }

    (first, second)
}

fn line_points(scene: Scene, count: usize) -> DataMatrix {
    let mut rng = DeterministicRng::new(SEED ^ 0x2345);
    let mut points = DataMatrix::zeros(count, 2);
    for index in 0..count {
        let x = rng.range(-20.0, 20.0);
        points.set(index, 0, x);
        points.set(
            index,
            1,
            if scene.is_outlier(index) {
                rng.range(-30.0, 30.0)
            } else {
                0.7 * x + 1.5 + rng.signed_noise(scene.noise() * 10.0)
            },
        );
    }
    points
}

fn plane_points(scene: Scene, count: usize) -> DataMatrix {
    let mut rng = DeterministicRng::new(SEED ^ 0x3456);
    let mut points = DataMatrix::zeros(count, 3);
    for index in 0..count {
        let x = rng.range(-20.0, 20.0);
        let y = rng.range(-20.0, 20.0);
        points.set(index, 0, x);
        points.set(index, 1, y);
        points.set(
            index,
            2,
            if scene.is_outlier(index) {
                rng.range(-30.0, 30.0)
            } else {
                0.25 * x - 0.4 * y + 3.0 + rng.signed_noise(scene.noise() * 10.0)
            },
        );
    }
    points
}

fn rigid_correspondences(scene: Scene, count: usize) -> (DataMatrix, DataMatrix) {
    let rotation = Rotation3::from_euler_angles(0.15, -0.2, 0.1);
    let translation = Vector3::new(3.0, -2.0, 1.0);
    let mut rng = DeterministicRng::new(SEED ^ 0x4567);
    let mut source = DataMatrix::zeros(count, 3);
    let mut target = DataMatrix::zeros(count, 3);
    for index in 0..count {
        let point = Vector3::new(
            rng.range(-10.0, 10.0),
            rng.range(-10.0, 10.0),
            rng.range(-10.0, 10.0),
        );
        let transformed = rotation * point + translation;
        for dim in 0..3 {
            source.set(index, dim, point[dim]);
            target.set(
                index,
                dim,
                if scene.is_outlier(index) {
                    rng.range(-20.0, 20.0)
                } else {
                    transformed[dim] + rng.signed_noise(scene.noise() * 10.0)
                },
            );
        }
    }
    (source, target)
}

fn absolute_pose_correspondences(scene: Scene, count: usize) -> (DataMatrix, DataMatrix) {
    let mut rng = DeterministicRng::new(SEED ^ 0x5678);
    let mut world = DataMatrix::zeros(count, 3);
    let mut image = DataMatrix::zeros(count, 2);
    for index in 0..count {
        let point = Vector3::new(
            rng.range(-2.0, 2.0),
            rng.range(-1.5, 1.5),
            rng.range(4.0, 8.0),
        );
        for dim in 0..3 {
            world.set(index, dim, point[dim]);
        }
        image.set(
            index,
            0,
            if scene.is_outlier(index) {
                rng.range(-1.0, 1.0)
            } else {
                point.x / point.z + rng.signed_noise(scene.noise())
            },
        );
        image.set(
            index,
            1,
            if scene.is_outlier(index) {
                rng.range(-1.0, 1.0)
            } else {
                point.y / point.z + rng.signed_noise(scene.noise())
            },
        );
    }
    (world, image)
}

fn benchmark_estimators(c: &mut Criterion) {
    let mut group = c.benchmark_group("public_api");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for scene in Scene::ALL {
        let seed = SEED ^ (scene as u64);
        let settings = settings(seed);
        let (source, target) = homography_correspondences(scene, 256);
        group.bench_with_input(
            BenchmarkId::new("homography", scene.name()),
            &(source, target, settings.clone()),
            |b, (source, target, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_homography(
                            black_box(source),
                            black_box(target),
                            0.5,
                            Some(settings.clone()),
                        )
                        .expect("benchmark scene must estimate a homography"),
                    )
                });
            },
        );

        let (source, target) = image_correspondences(scene, 256);
        group.bench_with_input(
            BenchmarkId::new("fundamental", scene.name()),
            &(source, target, settings.clone()),
            |b, (source, target, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_fundamental_matrix(
                            black_box(source),
                            black_box(target),
                            0.01,
                            Some(settings.clone()),
                        )
                        .expect("benchmark scene must estimate a fundamental matrix"),
                    )
                });
            },
        );

        let (source, target) = image_correspondences(scene, 256);
        group.bench_with_input(
            BenchmarkId::new("essential", scene.name()),
            &(source, target, settings.clone()),
            |b, (source, target, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_essential_matrix(
                            black_box(source),
                            black_box(target),
                            0.01,
                            Some(settings.clone()),
                        )
                        .expect("benchmark scene must estimate an essential matrix"),
                    )
                });
            },
        );

        let points = line_points(scene, 256);
        group.bench_with_input(
            BenchmarkId::new("line", scene.name()),
            &(points, settings.clone()),
            |b, (points, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_line(black_box(points), 0.05, Some(settings.clone()))
                            .expect("benchmark scene must estimate a line"),
                    )
                });
            },
        );

        let points = plane_points(scene, 256);
        group.bench_with_input(
            BenchmarkId::new("plane", scene.name()),
            &(points, settings.clone()),
            |b, (points, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_plane(black_box(points), 0.05, Some(settings.clone()))
                            .expect("benchmark scene must estimate a plane"),
                    )
                });
            },
        );

        let (source, target) = rigid_correspondences(scene, 256);
        group.bench_with_input(
            BenchmarkId::new("rigid_transform", scene.name()),
            &(source, target, settings.clone()),
            |b, (source, target, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_rigid_transform(
                            black_box(source),
                            black_box(target),
                            0.05,
                            Some(settings.clone()),
                        )
                        .expect("benchmark scene must estimate a rigid transform"),
                    )
                });
            },
        );

        let (world, image) = absolute_pose_correspondences(scene, 128);
        group.bench_with_input(
            BenchmarkId::new("absolute_pose", scene.name()),
            &(world, image, settings.clone()),
            |b, (world, image, settings)| {
                b.iter(|| {
                    black_box(
                        estimate_absolute_pose(
                            black_box(world),
                            black_box(image),
                            0.01,
                            Some(settings.clone()),
                        )
                        .expect("benchmark scene must estimate an absolute pose"),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_estimators);
criterion_main!(benches);
