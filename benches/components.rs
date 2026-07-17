//! Deterministic component-level benchmarks for CodSpeed regression tracking.
//!
//! The public API bench measures complete estimates. This target isolates the
//! selection and preprocessing components so a regression can be attributed to
//! a sampler, neighborhood construction, or preconditioner.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use inlier::{
    core::Sampler,
    models::Homography,
    preconditioner::{IdentityPreconditioner, Preconditioner},
    samplers::{
        AdaptiveReorderingSampler, DummyNeighborhood, GridNeighborhoodGraph, ImportanceSampler,
        KdTreeNeighborhoodGraph, NapsacSampler, NeighborhoodGraph, ProgressiveNapsacSampler,
        ProsacSampler, UniformRandomSampler,
    },
    types::DataMatrix,
};
use std::time::Duration;

const POINTS: usize = 1_024;
const SAMPLE_SIZE: usize = 8;
const SEED: u64 = 0x5EED_CAFE_D00D_BAAD;

fn point_data() -> DataMatrix {
    let mut data = DataMatrix::zeros(POINTS, 2);
    for index in 0..POINTS {
        let x = (index % 32) as f64 * 8.0 + (index % 7) as f64 * 0.1;
        let y = (index / 32) as f64 * 8.0 + (index % 11) as f64 * 0.1;
        data.set(index, 0, x);
        data.set(index, 1, y);
    }
    data
}

fn bench_sampler<S: Sampler>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    mut sampler: S,
    data: &DataMatrix,
) {
    group.bench_function(name, |bench| {
        let mut sample = [0; SAMPLE_SIZE];
        bench.iter(|| {
            assert!(sampler.sample(black_box(data), SAMPLE_SIZE, &mut sample));
            sampler.update(&sample, SAMPLE_SIZE, 0, 1.0);
            black_box(sample)
        });
    });
}

fn benchmark_components(c: &mut Criterion) {
    let data = point_data();
    let probabilities: Vec<f64> = (0..POINTS)
        .map(|index| 1.0 + (POINTS - index) as f64 / POINTS as f64)
        .collect();

    let mut samplers = c.benchmark_group("components/sampler_step");
    samplers.sample_size(20);
    samplers.warm_up_time(Duration::from_secs(1));
    samplers.measurement_time(Duration::from_secs(3));
    bench_sampler(
        &mut samplers,
        "uniform",
        UniformRandomSampler::new(Some(SEED)),
        &data,
    );
    bench_sampler(
        &mut samplers,
        "prosac",
        ProsacSampler::new(100_000, Some(SEED)),
        &data,
    );
    bench_sampler(
        &mut samplers,
        "importance",
        ImportanceSampler::from_probabilities_with_seed(&probabilities, SEED),
        &data,
    );
    bench_sampler(
        &mut samplers,
        "adaptive_reordering",
        AdaptiveReorderingSampler::new_with_seed(&probabilities, 0.9765, 0.01, SEED),
        &data,
    );
    bench_sampler(
        &mut samplers,
        "napsac_dummy_neighborhood",
        NapsacSampler::from_seed(SEED, DummyNeighborhood::new(POINTS, 16)),
        &data,
    );
    bench_sampler(
        &mut samplers,
        "progressive_napsac_dummy_neighborhood",
        ProgressiveNapsacSampler::from_seed(SEED, DummyNeighborhood::new(POINTS, 16), 10.0),
        &data,
    );
    samplers.finish();

    let mut neighborhoods = c.benchmark_group("components/neighborhood_build");
    neighborhoods.sample_size(20);
    neighborhoods.warm_up_time(Duration::from_secs(1));
    neighborhoods.measurement_time(Duration::from_secs(3));
    neighborhoods.bench_function("grid", |bench| {
        bench.iter(|| {
            let mut graph = GridNeighborhoodGraph::new(16.0, 16.0, 32);
            NeighborhoodGraph::initialize(&mut graph, black_box(&data));
            black_box(graph.neighbors(POINTS / 2).len())
        });
    });
    neighborhoods.bench_function("kdtree_2d", |bench| {
        bench.iter(|| {
            let mut graph = KdTreeNeighborhoodGraph::<2>::new(16);
            graph.initialize(black_box(&data));
            black_box(graph.neighbors(POINTS / 2).len())
        });
    });
    neighborhoods.finish();

    let mut preconditioners = c.benchmark_group("components/preconditioner");
    preconditioners.sample_size(20);
    preconditioners.warm_up_time(Duration::from_secs(1));
    preconditioners.measurement_time(Duration::from_secs(3));
    preconditioners.bench_function("identity_normalize", |bench| {
        let preconditioner = IdentityPreconditioner;
        bench.iter(|| {
            let (normalized, state) =
                <IdentityPreconditioner as Preconditioner<Homography>>::normalize(
                    &preconditioner,
                    black_box(&data),
                );
            black_box((normalized.n_points(), state))
        });
    });
    preconditioners.finish();
}

criterion_group!(benches, benchmark_components);
criterion_main!(benches);
