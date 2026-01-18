use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use inlier::{ColoredIcpPipeline, ColoredIcpScale, ColoredIcpSettings, PointCloud};
use nalgebra::{DMatrix, Matrix4};

fn make_grid_cloud(rows: usize, cols: usize, spacing: f64) -> PointCloud {
    let mut data = Vec::with_capacity(rows * cols * 6);
    for i in 0..rows {
        for j in 0..cols {
            let x = i as f64 * spacing;
            let y = j as f64 * spacing;
            let z = 0.0;
            let r = (i as f64) / (rows.max(1) as f64);
            let g = (j as f64) / (cols.max(1) as f64);
            let b = 0.5;
            data.extend_from_slice(&[x, y, z, r, g, b]);
        }
    }
    let mat = DMatrix::from_row_slice(rows * cols, 6, &data);
    PointCloud::from_xyzrgb(&mat).expect("valid xyzrgb cloud")
}

fn bench_colored_icp(c: &mut Criterion) {
    let source = make_grid_cloud(30, 30, 0.03);
    let target = make_grid_cloud(30, 30, 0.03);

    let scales = vec![ColoredIcpScale {
        voxel_radius: 0.08,
        max_iteration: 8,
    }];
    let settings = ColoredIcpSettings {
        distance_threshold: 0.15,
        normal_radius: 0.08,
        gradient_radius: 0.08,
        ..ColoredIcpSettings::default()
    };

    let mut group = c.benchmark_group("colored_icp");
    group.throughput(Throughput::Elements(source.len() as u64));
    group.bench_function("single_scale", |b| {
        b.iter(|| {
            let mut pipeline = ColoredIcpPipeline::new()
                .with_scales(scales.clone())
                .with_settings(settings.clone());
            let _ = pipeline
                .run(&source, &target, Matrix4::identity())
                .expect("registration should succeed");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_colored_icp);
criterion_main!(benches);
