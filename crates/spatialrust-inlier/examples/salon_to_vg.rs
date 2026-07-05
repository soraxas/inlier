//! Segment a point cloud with region-growing and export PolyFit `.vg` (vertex
//! groups), so PolyFit's Python binding (which does not do plane detection) can
//! reconstruct from it.
//!
//! Usage: salon_to_vg <input.ply> <output.vg>

use std::io::Write;

use spatialrust_inlier::convert::point_cloud_to_data_matrix;
use spatialrust_inlier::io::read_point_cloud_file;
use spatialrust_inlier::RansacMode;
use spatialrust_inlier::{GlobalPlanePeeling, PlaneEstimator, RegionGrowing};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map(String::as_str).unwrap_or("SALON.ply");
    let output = args.get(2).map(String::as_str).unwrap_or("SALON.vg");

    // Optional tuning args: voxel_div min_cluster dist_factor angle_deg
    let voxel_div: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500.0);
    let min_cluster: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let dist_factor: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.006);
    let angle_deg: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(15.0);

    let cloud = read_point_cloud_file(input)?;
    let dm = point_cloud_to_data_matrix(&cloud)?;
    let raw: Vec<[f32; 3]> = (0..dm.n_points())
        .map(|i| [dm.get(i, 0) as f32, dm.get(i, 1) as f32, dm.get(i, 2) as f32])
        .collect();
    eprintln!("read {} points from {input}", raw.len());

    // Scale-relative parameters from the bbox diagonal.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &raw {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    eprintln!("bbox diagonal = {diag:.4}");

    // Voxel-downsample (one centroid per cell) to even out density + speed up.
    let voxel = diag / voxel_div;
    let pts = voxel_downsample(&raw, voxel);
    eprintln!("downsampled to {} points (voxel = {voxel:.5})", pts.len());

    // Segment. Method arg (7): "rg" = region-growing, "peel" = global peeling.
    let method = args.get(7).map(String::as_str).unwrap_or("rg");
    let dist_thresh = diag * dist_factor;
    let planes = match method {
        "peel" => GlobalPlanePeeling {
            k: 20,
            normal_consensus: false,
            dist_thresh,
            angle_thresh: angle_deg.to_radians(),
            min_support: min_cluster,
            max_planes: 60,
            max_iterations: 1000,
            confidence: 0.99,
        }
        .estimate(&pts),
        _ => RegionGrowing {
            k: 20,
            angle_thresh: angle_deg.to_radians(),
            min_cluster_size: min_cluster,
            dist_thresh,
            mode: RansacMode::Magsac,
            sigma_max: (dist_thresh * 1.5) as f64,
            max_iterations: 1000,
            confidence: 0.99,
        }
        .estimate(&pts),
    };
    eprintln!("method={method}, found {} planes", planes.len());

    write_vg(output, &pts, &planes)?;
    eprintln!("wrote {output}");
    Ok(())
}

/// Keep one centroid per occupied voxel cell.
fn voxel_downsample(pts: &[[f32; 3]], voxel: f32) -> Vec<[f32; 3]> {
    use std::collections::HashMap;
    let key = |p: &[f32; 3]| {
        (
            (p[0] / voxel).floor() as i64,
            (p[1] / voxel).floor() as i64,
            (p[2] / voxel).floor() as i64,
        )
    };
    let mut acc: HashMap<(i64, i64, i64), ([f64; 3], u32)> = HashMap::new();
    for p in pts {
        let e = acc.entry(key(p)).or_insert(([0.0; 3], 0));
        for k in 0..3 {
            e.0[k] += p[k] as f64;
        }
        e.1 += 1;
    }
    acc.into_values()
        .map(|(s, n)| [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32])
        .collect()
}

fn write_vg(
    path: &str,
    pts: &[[f32; 3]],
    planes: &[([f32; 3], f32, Vec<usize>)],
) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);

    write!(w, "num_points: {}\n", pts.len())?;
    for p in pts {
        write!(w, "{} {} {} ", p[0], p[1], p[2])?;
    }
    writeln!(w)?;
    writeln!(w, "num_colors: 0")?;
    writeln!(w, "num_normals: 0")?;
    writeln!(w, "num_groups: {}", planes.len())?;

    let palette = [
        [0.9f32, 0.3, 0.2], [0.2, 0.7, 0.9], [0.3, 0.85, 0.3], [0.9, 0.75, 0.1],
        [0.7, 0.3, 0.9], [0.9, 0.2, 0.55], [0.2, 0.6, 0.5], [0.6, 0.6, 0.2],
    ];
    for (gi, (n, d, idx)) in planes.iter().enumerate() {
        let c = palette[gi % palette.len()];
        writeln!(w, "group_type: 0")?;
        writeln!(w, "num_group_parameters: 4")?;
        writeln!(w, "group_parameters: {} {} {} {}", n[0], n[1], n[2], d)?;
        writeln!(w, "group_label: plane_{gi}")?;
        writeln!(w, "group_color: {} {} {}", c[0], c[1], c[2])?;
        write!(w, "group_num_point: {}\n", idx.len())?;
        for &i in idx {
            write!(w, "{} ", i)?;
        }
        writeln!(w)?;
        writeln!(w, "num_children: 0")?;
    }
    w.flush()
}
