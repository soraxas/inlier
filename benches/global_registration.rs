use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use inlier::pointcloud::NormalEstimationStrategy;
use inlier::{
    FeatureMatchSettings, FpfhSettings, GlobalRegistrationSettings, PointCloud,
    compute_fpfh_features, registration_fgr_based_on_correspondence,
    registration_ransac_based_on_correspondence,
};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

fn bench_global_registration(c: &mut Criterion) {
    let source = make_sphere_cloud(400);
    let target = make_sphere_cloud(400);

    let normal_estimator = inlier::pointcloud::KnnNormalEstimator;
    let source = normal_estimator
        .estimate(&source, 0.1, 30)
        .expect("normals");
    let target = normal_estimator
        .estimate(&target, 0.1, 30)
        .expect("normals");

    let fpfh_settings = FpfhSettings {
        radius: 0.25,
        max_nn: 100,
    };
    let source_fpfh = compute_fpfh_features(&source, &fpfh_settings).expect("fpfh");
    let target_fpfh = compute_fpfh_features(&target, &fpfh_settings).expect("fpfh");

    let match_settings = FeatureMatchSettings {
        mutual: false,
        max_correspondences: 2000,
    };
    let correspondences =
        inlier::match_features(&source_fpfh, &target_fpfh, &match_settings).expect("match");

    let mut group = c.benchmark_group("global_registration");
    group.throughput(Throughput::Elements(source.len() as u64));

    group.bench_function("ransac_correspondence", |b| {
        b.iter(|| {
            let settings = GlobalRegistrationSettings {
                distance_threshold: 0.08,
                max_iterations: 500,
                checkers: Vec::new(),
                ..GlobalRegistrationSettings::default()
            };
            let _ = registration_ransac_based_on_correspondence(
                &source,
                &target,
                &correspondences,
                &settings,
            )
            .expect("ransac");
        });
    });

    group.bench_function("fgr_correspondence", |b| {
        b.iter(|| {
            let options = inlier::FastGlobalRegistrationOptions {
                maximum_correspondence_distance: 0.08,
                max_iterations: 32,
                ..inlier::FastGlobalRegistrationOptions::default()
            };
            let _ = registration_fgr_based_on_correspondence(
                &source,
                &target,
                &correspondences,
                &options,
            )
            .expect("fgr");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_global_registration);
criterion_main!(benches);
