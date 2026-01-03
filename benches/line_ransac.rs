use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inlier::api::estimate_line;
use inlier::settings::{
    InlierSelectorType, LocalOptimizationType, NeighborhoodType, RansacSettings, SamplerType,
    ScoringType, TerminationType,
};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[allow(clippy::field_reassign_with_default)]
fn make_settings(threshold: f64) -> RansacSettings {
    let mut settings = RansacSettings::default();
    settings.min_iterations = 200;
    settings.max_iterations = 800;
    settings.inlier_threshold = threshold;
    settings.scoring = ScoringType::Ransac;
    settings.sampler = SamplerType::Uniform;
    settings.neighborhood = NeighborhoodType::BruteForce;
    settings.local_optimization = LocalOptimizationType::None;
    settings.final_optimization = LocalOptimizationType::None;
    settings.inlier_selector = InlierSelectorType::None;
    settings.termination_criterion = TerminationType::Ransac;
    settings
}

fn generate_line_data(n_inliers: usize, n_outliers: usize) -> DMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(1337);
    let total = n_inliers + n_outliers;
    let mut data = Vec::with_capacity(total * 2);

    for _ in 0..n_inliers {
        let x = rng.random_range(-100.0..100.0);
        let noise = rng.random_range(-0.5..0.5);
        let y = 2.0 * x + 1.0 + noise;
        data.push(x);
        data.push(y);
    }

    for _ in 0..n_outliers {
        let x = rng.random_range(-100.0..100.0);
        let y = rng.random_range(-100.0..100.0);
        data.push(x);
        data.push(y);
    }

    DMatrix::from_row_slice(total, 2, &data)
}

fn bench_estimate_line(c: &mut Criterion) {
    let threshold = 1.5;
    let settings = make_settings(threshold);
    let mut group = c.benchmark_group("estimate_line");

    for (name, inliers, outliers) in [
        ("small", 500_usize, 200_usize),
        ("medium", 2500_usize, 1000_usize),
    ] {
        let data = generate_line_data(inliers, outliers);
        group.throughput(Throughput::Elements(data.nrows() as u64));
        group.bench_with_input(BenchmarkId::new("points", name), &data, |b, data| {
            b.iter(|| {
                let result = estimate_line(data, threshold, Some(settings.clone()));
                result.expect("estimate_line should succeed for synthetic data");
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_estimate_line);
criterion_main!(benches);
