use criterion::{Criterion, black_box, criterion_group, criterion_main};
use inlier::presets::pointcloud_registration::adaptive_scale_voting;
use inlier::types::DataMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load a CSV file where each row is a dimension (3 rows for x,y,z)
/// and each column is a point. Returns a DataMatrix (N points x 3 dims).
fn load_csv_points(path: &str) -> Result<DataMatrix, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read all lines
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let values: Result<Vec<f64>, _> =
            line.split(',').map(|s| s.trim().parse::<f64>()).collect();
        rows.push(values?);
    }

    if rows.is_empty() {
        return Err("No data in CSV".into());
    }

    // CSV has dimensions as rows, points as columns
    // We need to transpose: N points x D dimensions
    let n_dims = rows.len();
    let n_points = rows[0].len();

    let mut flat = Vec::with_capacity(n_points * n_dims);
    for point_idx in 0..n_points {
        for dim_idx in 0..n_dims {
            flat.push(rows[dim_idx][point_idx]);
        }
    }

    Ok(DataMatrix::from_row_slice(n_points, n_dims, &flat))
}

/// Create correspondences matrix from source and target points
fn create_correspondences(src: &DataMatrix, dst: &DataMatrix) -> DataMatrix {
    assert_eq!(src.n_points(), dst.n_points());
    assert_eq!(src.n_dims(), 3);
    assert_eq!(dst.n_dims(), 3);

    let n = src.n_points();
    let mut data = DataMatrix::zeros(n, 6);

    for i in 0..n {
        data.set(i, 0, src.get(i, 0));
        data.set(i, 1, src.get(i, 1));
        data.set(i, 2, src.get(i, 2));
        data.set(i, 3, dst.get(i, 0));
        data.set(i, 4, dst.get(i, 1));
        data.set(i, 5, dst.get(i, 2));
    }

    data
}

fn bench_teaser_scale_estimation(c: &mut Criterion) {
    // Load TEASER++ test data
    let src_path = "TEASER-plusplus/test/teaser/data/registration_test/objectIn.csv";
    let dst_path = "TEASER-plusplus/test/teaser/data/registration_test/sceneIn.csv";

    if !std::path::Path::new(src_path).exists() {
        eprintln!("Skipping benchmark: test data not found");
        return;
    }

    let src = load_csv_points(src_path).expect("Failed to load source points");
    let dst = load_csv_points(dst_path).expect("Failed to load target points");
    let data = create_correspondences(&src, &dst);

    // TEASER++ reference parameters
    let noise_bound = 0.0067364;
    let c_bar = 1.0;
    let max_pairs = 100_000;

    // Configure for slower benchmarks: reduce samples, increase time
    let mut group = c.benchmark_group("teaser_scale");
    group.sample_size(10); // Only 10 samples instead of 100
    group.measurement_time(std::time::Duration::from_secs(20)); // 20s measurement
    group.warm_up_time(std::time::Duration::from_secs(3));

    group.bench_function("estimation_168pts", |b| {
        b.iter(|| {
            adaptive_scale_voting(
                black_box(&data),
                black_box(noise_bound),
                black_box(c_bar),
                black_box(max_pairs),
            )
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)  // Reduce default sample size globally
        .warm_up_time(std::time::Duration::from_secs(3));
    targets = bench_teaser_scale_estimation
}
criterion_main!(benches);
