#![no_main]

use inlier::{
    estimate_absolute_pose, estimate_essential_matrix, estimate_fundamental_matrix,
    estimate_homography, estimate_line, estimate_plane, estimate_rigid_transform,
    settings::MetasacSettings, types::DataMatrix,
};
use libfuzzer_sys::fuzz_target;

fn values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .take(64)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect()
}

fn fill_matrix(values: &[f64], columns: usize, offset: usize) -> DataMatrix {
    let mut matrix = DataMatrix::zeros(8, columns);
    for row in 0..8 {
        for column in 0..columns {
            matrix.set(
                row,
                column,
                values[(offset + row * columns + column) % values.len()],
            );
        }
    }
    matrix
}

fuzz_target!(|data: &[u8]| {
    let values = values(data);
    if values.is_empty() {
        return;
    }
    let points1 = fill_matrix(&values, 2, 0);
    let points2 = fill_matrix(&values, 2, 16);
    let points3d = fill_matrix(&values, 3, 32);
    let targets3d = fill_matrix(&values, 3, 8);
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
    if let Ok(result) = estimate_fundamental_matrix(&points1, &points2, 1.0, Some(settings.clone()))
    {
        assert!(result.model.f.iter().all(|value| value.is_finite()));
    }
    if let Ok(result) = estimate_essential_matrix(&points1, &points2, 1.0, Some(settings.clone())) {
        assert!(result.model.e.iter().all(|value| value.is_finite()));
    }
    if let Ok(result) = estimate_absolute_pose(&points3d, &points1, 1.0, Some(settings.clone())) {
        assert!(result
            .model
            .rotation
            .coords
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .model
            .translation
            .vector
            .iter()
            .all(|value| value.is_finite()));
    }
    if let Ok(result) = estimate_line(&points1, 1.0, Some(settings.clone())) {
        assert!(result.model.params.iter().all(|value| value.is_finite()));
    }
    if let Ok(result) = estimate_plane(&points3d, 1.0, Some(settings.clone())) {
        assert!(result.model.normal.iter().all(|value| value.is_finite()));
        assert!(result.model.d.is_finite());
    }
    if let Ok(result) = estimate_rigid_transform(&points3d, &targets3d, 1.0, Some(settings)) {
        assert!(result
            .model
            .rotation
            .coords
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .model
            .translation
            .vector
            .iter()
            .all(|value| value.is_finite()));
    }
});
