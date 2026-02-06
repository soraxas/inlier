//! Example: Build a RANSAC pipeline from scratch with custom Rust components.
//!
//! Mirrors the Python example at `examples/python/pipeline_scratch_linear_regression.py`,
//! but implemented directly against the Rust `core` traits and `CorePipeline`.

use inlier::core::{Estimator, NoopInlierSelector, RansacTerminationCriterion, Sampler, Scoring};
use inlier::optimisers::NoopLocalOptimizer;
use inlier::pipeline::CorePipeline;
use inlier::preconditioner::{IdentityPreconditioner, Preconditioner};
use inlier::scoring::Score;
use inlier::settings::MetasacSettings;
use inlier::types::DataMatrix;
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Simple line model in slope-intercept form.
#[derive(Clone, Debug)]
struct LineModel {
    m: f64,
    b: f64,
}

struct RandomSampler {
    rng: rand::rngs::ThreadRng,
}

impl RandomSampler {
    fn new() -> Self {
        Self { rng: rand::rng() }
    }
}

#[cfg_attr(feature = "hotpath-prof", hotpath::measure_all)]
impl Sampler for RandomSampler {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
        if data.nrows() < sample_size || out_indices.len() < sample_size {
            return false;
        }

        let mut indices: Vec<usize> = (0..data.nrows()).collect();
        indices.shuffle(&mut self.rng);
        out_indices[..sample_size].copy_from_slice(&indices[..sample_size]);
        true
    }

    fn update(
        &mut self,
        _sample: &[usize],
        _sample_size: usize,
        _iteration: usize,
        _score_hint: f64,
    ) {
    }
}

struct LineEstimator {
    sample_size: usize,
}

impl LineEstimator {
    fn new(rng: &mut impl Rng) -> Self {
        // Randomly choose 2 or 3 points per sample to mirror the Python example.
        let sample_size = rng.random_range(2..=3);
        Self { sample_size }
    }
}

#[cfg_attr(feature = "hotpath-prof", hotpath::measure_all)]
impl Estimator for LineEstimator {
    type Model = LineModel;

    fn sample_size(&self) -> usize {
        self.sample_size
    }

    fn non_minimal_sample_size(&self) -> usize {
        self.sample_size
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        if sample.len() < 2 {
            return false;
        }
        let (x1, _y1) = (data[(sample[0], 0)], data[(sample[0], 1)]);
        let (x2, _y2) = (data[(sample[1], 0)], data[(sample[1], 1)]);
        (x2 - x1).abs() > 1e-9
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        if sample.len() < 2 {
            return Vec::new();
        }

        let (x1, y1) = (data[(sample[0], 0)], data[(sample[0], 1)]);
        let (x2, y2) = (data[(sample[1], 0)], data[(sample[1], 1)]);
        let m = (y2 - y1) / (x2 - x1);
        let b = y1 - m * x1;
        vec![LineModel { m, b }]
    }

    fn is_valid_model(
        &self,
        _model: &Self::Model,
        _data: &DataMatrix,
        _sample: &[usize],
        _threshold: f64,
    ) -> bool {
        true
    }
}

#[derive(Clone)]
struct LineScoring {
    tau: f64,
}

#[cfg_attr(feature = "hotpath-prof", hotpath::measure_all)]
impl Scoring<LineModel> for LineScoring {
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.tau
    }

    fn score(
        &self,
        data: &DataMatrix,
        model: &LineModel,
        inliers_out: &mut Vec<usize>,
    ) -> Self::Score {
        inliers_out.clear();

        for i in 0..data.nrows() {
            let x = data[(i, 0)];
            let y = data[(i, 1)];
            let residual = (y - (model.m * x + model.b)).abs();
            if residual <= self.tau {
                inliers_out.push(i);
            }
        }

        let inlier_count = inliers_out.len();
        // Prefer higher inlier counts; value can simply mirror the count.
        Score::new(inlier_count, inlier_count as f64)
    }
}

#[hotpath::main]
#[cfg_attr(feature = "hotpath-prof", hotpath::measure)]
fn main() {
    let mut rng = rand::rng();
    let n_sample: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let seed: u32 = rng.random_range(0..u32::MAX);
    let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);

    let slope_gt = seeded_rng.random_range(-10.0..10.0);
    let intercept_gt = seeded_rng.random_range(-10.0..10.0);
    let noise_sigma = seeded_rng.random_range(0.0..0.5);

    let xs: Vec<f64> = (0..n_sample)
        .map(|i| -1.0 + 2.0 * (i as f64) / (n_sample as f64 - 1.0))
        .collect();
    let mut ys = Vec::with_capacity(n_sample);
    for &x in &xs {
        let noise: f64 = seeded_rng.random_range(-noise_sigma..noise_sigma);
        ys.push(slope_gt * x + intercept_gt + noise);
    }

    let mut data_flat = Vec::with_capacity(n_sample * 2);
    for (&x, &y) in xs.iter().zip(&ys) {
        data_flat.push(x);
        data_flat.push(y);
    }
    let data = DMatrix::from_row_slice(n_sample, 2, &data_flat);

    let settings = MetasacSettings {
        min_iterations: 500,
        max_iterations: 2000,
        confidence: 0.999,
        ..MetasacSettings::default()
    };

    let estimator = LineEstimator::new(&mut seeded_rng);
    let sampler = RandomSampler::new();
    let scoring = LineScoring { tau: 0.1 };
    let termination = RansacTerminationCriterion {
        confidence: settings.confidence,
    };

    let pipeline = CorePipeline::<
        LineEstimator,
        RandomSampler,
        LineScoring,
        NoopLocalOptimizer,
        RansacTerminationCriterion,
        NoopInlierSelector,
        IdentityPreconditioner,
    >::new(settings, estimator, sampler, scoring, termination)
    .with_inlier_selector(NoopInlierSelector)
    .with_local_optimizer(NoopLocalOptimizer)
    .with_preconditioner(IdentityPreconditioner);
    let result = pipeline.run(&data).expect("RANSAC failed");

    let slope_err = (result.model.m - slope_gt).abs() / slope_gt.abs().max(1e-9);
    let intercept_err = (result.model.b - intercept_gt).abs() / intercept_gt.abs().max(1e-9);

    println!("Seed: {seed}");
    println!(
        "Ground truth: slope={slope_gt:.4}, intercept={intercept_gt:.4}, noise_sigma={noise_sigma:.4}"
    );
    println!(
        "Estimated:    slope={:.4}, intercept={:.4}",
        result.model.m, result.model.b
    );
    println!(
        "Relative error: slope={:.4}%, intercept={:.4}%",
        slope_err * 100.0,
        intercept_err * 100.0
    );
    println!(
        "Inliers: {} / {}, Score: {:.3}, Iterations: {}",
        result.inliers.len(),
        n_sample,
        result.score.value,
        result.iterations
    );
}
