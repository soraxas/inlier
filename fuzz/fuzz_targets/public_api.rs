#![no_main]

use inlier::{
    estimate_essential_matrix, estimate_fundamental_matrix, estimate_homography,
    settings::MetasacSettings,
    types::DataMatrix,
};
use libfuzzer_sys::fuzz_target;

fn values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .take(16)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let values = values(data);
    if values.len() < 16 {
        return;
    }
    let mut points1 = DataMatrix::zeros(8, 2);
    let mut points2 = DataMatrix::zeros(8, 2);
    for index in 0..8 {
        points1.set(index, 0, values[index * 2]);
        points1.set(index, 1, values[index * 2 + 1]);
        points2.set(index, 0, values[(index * 2 + 8) % values.len()]);
        points2.set(index, 1, values[(index * 2 + 9) % values.len()]);
    }
    let settings = MetasacSettings {
        min_iterations: 1,
        max_iterations: 1,
        max_sampling_attempts: 1,
        rng_seed: Some(0),
        ..MetasacSettings::default()
    };

    if let Ok(result) = estimate_homography(&points1, &points2, 1.0, Some(settings.clone())) {
        assert!(result.model.h.iter().all(|value| value.is_finite()));
    }
    if let Ok(result) = estimate_fundamental_matrix(&points1, &points2, 1.0, Some(settings.clone())) {
        assert!(result.model.f.iter().all(|value| value.is_finite()));
    }
    if let Ok(result) = estimate_essential_matrix(&points1, &points2, 1.0, Some(settings)) {
        assert!(result.model.e.iter().all(|value| value.is_finite()));
    }
});
