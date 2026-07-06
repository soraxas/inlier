//! Full building pipeline with per-stage inspection dumps.
//!
//! Usage: inspect_building <input.ply> <outdir>
//!
//! Writes four PolyFit `.vg` files into <outdir>, coloured for `render6.py`:
//!   01_original — raw cloud (original coords; shows any tilt)
//!   02_aligned  — gravity/Manhattan-aligned coords (up → Z)
//!   03_storeys  — aligned, coloured per detected storey
//!   04_walls    — aligned, per-floor walls (red = exterior, blue = interior)

use std::io::Write;

use spatialrust_inlier::auto_tune::auto_tune_settings;
use spatialrust_inlier::convert::point_cloud_to_data_matrix;
use spatialrust_inlier::io::read_point_cloud_file;
use spatialrust_inlier::plane_ops::merge_planes;
use spatialrust_inlier::{
    assign_storeys_columnwise, build_footprint2d, compute_normals, estimate_frame_from_normals,
    find_storeys, refine_up_from_normals, smooth_storey_labels, ManhattanPlanes, PlaneEstimator,
};

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt().max(1e-9);
    [a[0] / n, a[1] / n, a[2] / n]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map(String::as_str).unwrap_or("SALON.ply");
    let outdir = args.get(2).map(String::as_str).unwrap_or("inspect-out");
    std::fs::create_dir_all(outdir)?;

    // Read + voxel-downsample.
    let cloud = read_point_cloud_file(input)?;
    let dm = point_cloud_to_data_matrix(&cloud)?;
    let raw: Vec<[f32; 3]> = (0..dm.n_points())
        .map(|i| [dm.get(i, 0) as f32, dm.get(i, 1) as f32, dm.get(i, 2) as f32])
        .collect();
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &raw {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let pts = voxel_downsample(&raw, diag / 700.0);
    eprintln!("{} points ({} raw), diag {diag:.3}", pts.len(), raw.len());

    // Stage 1: original.
    let all: Vec<usize> = (0..pts.len()).collect();
    write_vg(&format!("{outdir}/01_original.vg"), &pts, &[([0.6, 0.6, 0.6], &all)])?;

    // Align: gravity + Manhattan frame. up = consensus normal of horizontal
    // surfaces (floors/ceilings), which makes the floors actually flat.
    let normals = compute_normals(&pts, 20);
    let frame = estimate_frame_from_normals(&normals);
    let up = refine_up_from_normals(&normals, frame.up);
    let h1 = unit([
        frame.h1[0] - up[0] * dot(frame.h1, up),
        frame.h1[1] - up[1] * dot(frame.h1, up),
        frame.h1[2] - up[2] * dot(frame.h1, up),
    ]);
    let h2 = unit(cross(up, h1));
    eprintln!("up={up:?} h1={h1:?} h2={h2:?}");
    // Centroid, then aligned coords (h1→x, h2→y, up→z).
    let c = {
        let mut s = [0.0f32; 3];
        for p in &pts {
            for k in 0..3 {
                s[k] += p[k];
            }
        }
        [s[0] / pts.len() as f32, s[1] / pts.len() as f32, s[2] / pts.len() as f32]
    };
    let aligned: Vec<[f32; 3]> = pts
        .iter()
        .map(|p| {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            [dot(d, h1), dot(d, h2), dot(d, up)]
        })
        .collect();

    // Stage 2: aligned.
    write_vg(&format!("{outdir}/02_aligned.vg"), &aligned, &[([0.6, 0.6, 0.6], &all)])?;

    // Split into storeys, then refine to per-column (2.5-D) labels so a
    // ground-only column can't leak into the upper storey.
    let storeys = find_storeys(&pts, up, 0.25, diag * 0.05);
    eprintln!("{} storeys: {storeys:?}", storeys.len());
    let labels0 = assign_storeys_columnwise(&pts, up, h1, h2, &storeys, diag * 0.03, 0.15);
    // Region-growing-style smoothing to flip isolated boundary points.
    let labels = smooth_storey_labels(&pts, &labels0, 16, 2);
    let palette = [[0.9, 0.4, 0.2], [0.2, 0.6, 0.9], [0.3, 0.85, 0.3], [0.8, 0.3, 0.9]];
    let mut per_storey_idx: Vec<Vec<usize>> = vec![Vec::new(); storeys.len().max(1)];
    for (i, &s) in labels.iter().enumerate() {
        per_storey_idx[s].push(i);
    }
    let storey_groups: Vec<([f32; 3], Vec<usize>)> = per_storey_idx
        .iter()
        .enumerate()
        .map(|(si, idx)| (palette[si % palette.len()], idx.clone()))
        .collect();
    let sg: Vec<([f32; 3], &[usize])> =
        storey_groups.iter().map(|(c, v)| (*c, v.as_slice())).collect();
    write_vg(&format!("{outdir}/03_storeys.vg"), &aligned, &sg)?;

    // Per-floor walls + exterior classification.
    // Data-adaptive parameters from measured noise σ + point spacing (replaces
    // the earlier diag-relative guesses; generalises across densities/sensors).
    let tuned = auto_tune_settings(&pts);
    eprintln!("{}", tuned.description);
    let expand = diag * 0.01; // slab half-thickness for coverage fill
    let fp_cell = diag * 0.015; // footprint occupancy cell (a few point-spacings)

    let mut wall_groups: Vec<([f32; 3], Vec<usize>)> = Vec::new();
    for (si, storey_idx) in per_storey_idx.iter().enumerate() {
        let sub: Vec<[f32; 3]> = storey_idx.iter().map(|&i| pts[i]).collect();
        let planes = ManhattanPlanes {
            k: 20,
            dist_thresh: tuned.dist_thresh,
            // Orientation-vote tolerance: keep it loose (region-vote, not region-
            // grow) but let noisier clouds widen it.
            angle_thresh: tuned.angle_thresh.max(30.0).to_radians(),
            min_support: tuned.min_cluster_size,
        }
        .estimate(&sub);
        // Keep walls (normal ⟂ up), remapped to global indices.
        let raw: Vec<([f32; 3], f32, Vec<usize>)> = planes
            .into_iter()
            .filter(|(n, _, _)| dot(*n, up).abs() < 0.5)
            .map(|(n, d, local)| (n, d, local.iter().map(|&li| storey_idx[li]).collect()))
            .collect();
        // Coplanar-merge: consolidate the parallel wall bands (same normal, close
        // offset) that MSAC's thin dist_thresh splits into slivers. Distinct
        // walls (front/back) are far apart in offset so stay separate.
        let walls = merge_planes(
            &raw,
            &pts,
            tuned.merge_angle_thresh.to_radians(),
            tuned.merge_dist_thresh,
            tuned.merge_min_pts,
        );

        // Exterior via a footprint flood-fill: build this storey's 2-D occupancy
        // grid (aligned x=h1, y=h2), flood the exterior empty space from the grid
        // border. A wall is exterior iff it borders that exterior region;
        // interior partitions and enclosed-courtyard walls do not. Handles
        // concave/L-shaped footprints, which the old side-test could not.
        let fp_pts: Vec<(f32, f32)> =
            storey_idx.iter().map(|&g| (aligned[g][0], aligned[g][1])).collect();
        // No morphological closing by default: on these all-points grids it made
        // outer scan-noise block the flood and lost real exterior walls. Closing
        // (>0) is available for cleaner clouds where gap-leaks dominate instead.
        let fp = build_footprint2d(&fp_pts, fp_cell, 0);

        let (mut ne, mut ni) = (0, 0);
        let raw_n = raw.len();
        for (n, d, gidx0) in walls {
            // In-plane direction t = up × n, and this wall's extent from the
            // confident merged inliers.
            let t = unit(cross(up, n));
            let (mut tlo, mut thi) = (f32::MAX, f32::MIN);
            for &g in &gidx0 {
                let tp = dot(pts[g], t);
                tlo = tlo.min(tp);
                thi = thi.max(tp);
            }
            // Slab-expand: absorb every storey point within `expand` of the wall
            // plane AND inside its footprint span, so the wall renders full
            // (fills the gray points RANSAC left out).
            let gidx: Vec<usize> = storey_idx
                .iter()
                .copied()
                .filter(|&g| {
                    (dot(n, pts[g]) + d).abs() < expand && {
                        let tp = dot(pts[g], t);
                        tp >= tlo - expand && tp <= thi + expand
                    }
                })
                .collect();
            // Footprint boundary test: project the wall's points to aligned 2-D,
            // and its normal to (h1, h2).
            let wall2d: Vec<(f32, f32)> =
                gidx.iter().map(|&g| (aligned[g][0], aligned[g][1])).collect();
            let (a, b) = (dot(n, h1), dot(n, h2));
            let m = (a * a + b * b).sqrt().max(1e-9);
            let exterior = fp.wall_is_exterior(&wall2d, (a / m, b / m));
            let color = if exterior {
                ne += 1;
                [0.92, 0.22, 0.22] // exterior red
            } else {
                ni += 1;
                [0.30, 0.60, 0.92] // interior blue
            };
            wall_groups.push((color, gidx));
        }
        eprintln!(
            "  storey {si}: {} walls after merge (from {raw_n}) ({ne} exterior, {ni} interior)",
            ne + ni
        );
    }
    let wg: Vec<([f32; 3], &[usize])> =
        wall_groups.iter().map(|(c, v)| (*c, v.as_slice())).collect();
    write_vg(&format!("{outdir}/04_walls.vg"), &aligned, &wg)?;

    eprintln!("wrote 4 stages to {outdir}/");
    Ok(())
}

fn voxel_downsample(pts: &[[f32; 3]], voxel: f32) -> Vec<[f32; 3]> {
    use std::collections::HashMap;
    let key = |p: &[f32; 3]| {
        ((p[0] / voxel).floor() as i64, (p[1] / voxel).floor() as i64, (p[2] / voxel).floor() as i64)
    };
    // Preserve first-insertion (scan) order so the output is deterministic:
    // HashMap iteration order is randomised per run, and any point-order change
    // ripples through kNN ties, clustering, and merges, making results
    // non-reproducible. (Sorting by voxel key is also deterministic but slabs
    // the points spatially, which starves the per-axis plane clustering.)
    let mut slot: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut cells: Vec<([f64; 3], u32)> = Vec::new();
    for p in pts {
        let idx = *slot.entry(key(p)).or_insert_with(|| {
            cells.push(([0.0; 3], 0));
            cells.len() - 1
        });
        for k in 0..3 {
            cells[idx].0[k] += p[k] as f64;
        }
        cells[idx].1 += 1;
    }
    cells
        .into_iter()
        .map(|(s, n)| {
            [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32]
        })
        .collect()
}

/// Write a PolyFit ASCII .vg: points + one group per `(color, indices)`.
fn write_vg(path: &str, pts: &[[f32; 3]], groups: &[([f32; 3], &[usize])]) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "num_points: {}", pts.len())?;
    for p in pts {
        write!(w, "{} {} {} ", p[0], p[1], p[2])?;
    }
    writeln!(w)?;
    writeln!(w, "num_colors: 0")?;
    writeln!(w, "num_normals: 0")?;
    writeln!(w, "num_groups: {}", groups.len())?;
    for (gi, (color, idx)) in groups.iter().enumerate() {
        writeln!(w, "group_type: 0")?;
        writeln!(w, "num_group_parameters: 4")?;
        writeln!(w, "group_parameters: 0 0 1 0")?;
        writeln!(w, "group_label: g{gi}")?;
        writeln!(w, "group_color: {} {} {}", color[0], color[1], color[2])?;
        writeln!(w, "group_num_point: {}", idx.len())?;
        for &i in idx.iter() {
            write!(w, "{} ", i)?;
        }
        writeln!(w)?;
        writeln!(w, "num_children: 0")?;
    }
    w.flush()
}
