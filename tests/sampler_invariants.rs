use inlier::{
    core::Sampler,
    samplers::{ProsacSampler, UniformRandomSampler},
    types::DataMatrix,
};

fn data(point_count: usize) -> DataMatrix {
    DataMatrix::zeros(point_count, 2)
}

fn assert_valid_sample(sample: &[usize], point_count: usize) {
    assert!(sample.iter().all(|&index| index < point_count));
    let mut sorted = sample.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), sample.len(), "sample must not repeat indices");
}

#[test]
fn uniform_sampler_is_seeded_unique_and_in_bounds() {
    let data = data(32);
    let mut first = UniformRandomSampler::new(Some(0xA11CE));
    let mut second = UniformRandomSampler::new(Some(0xA11CE));

    for _ in 0..32 {
        let mut left = [usize::MAX; 7];
        let mut right = [usize::MAX; 7];
        assert!(first.sample(&data, 7, &mut left));
        assert!(second.sample(&data, 7, &mut right));
        assert_eq!(left, right, "a fixed seed must be reproducible");
        assert_valid_sample(&left, data.n_points());
    }
}

#[test]
fn prosac_starts_from_the_high_priority_prefix_and_remains_valid() {
    let data = data(24);
    let mut sampler = ProsacSampler::new(8, Some(0xBADC0DE));

    let mut first = [usize::MAX; 5];
    assert!(sampler.sample(&data, 5, &mut first));
    assert_valid_sample(&first, data.n_points());
    assert!(first.iter().all(|&index| index < 5));

    for _ in 0..24 {
        let mut sample = [usize::MAX; 5];
        assert!(sampler.sample(&data, 5, &mut sample));
        assert_valid_sample(&sample, data.n_points());
    }
}

#[test]
fn samplers_reject_impossible_requests() {
    let data = data(3);
    let mut uniform = UniformRandomSampler::new(Some(1));
    let mut prosac = ProsacSampler::new(8, Some(1));
    let mut output = [0; 3];

    for sampler in [
        &mut uniform as &mut dyn Sampler,
        &mut prosac as &mut dyn Sampler,
    ] {
        assert!(!sampler.sample(&data, 0, &mut output));
        assert!(!sampler.sample(&data, 4, &mut output));
    }
}
