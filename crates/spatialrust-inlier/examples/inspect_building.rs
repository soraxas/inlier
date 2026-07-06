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

use spatialrust_inlier::convert::point_cloud_to_data_matrix;
use spatialrust_inlier::io::read_point_cloud_file;
use spatialrust_inlier::plane_ops::merge_planes;
use spatialrust_inlier::{
    assign_storeys_columnwise, compute_normals, estimate_frame_from_normals, find_storeys,
    refine_up_from_normals, ManhattanPlanes, PlaneEstimator,
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
    let labels = assign_storeys_columnwise(&pts, up, h1, h2, &storeys, diag * 0.03, 0.15);
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
    let mut wall_groups: Vec<([f32; 3], Vec<usize>)> = Vec::new();
    for (si, storey_idx) in per_storey_idx.iter().enumerate() {
        let sub: Vec<[f32; 3]> = storey_idx.iter().map(|&i| pts[i]).collect();
        let planes = ManhattanPlanes {
            k: 20,
            dist_thresh: diag * 0.006,
            angle_thresh: 35f32.to_radians(),
            min_support: 300,
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
        let walls = merge_planes(&raw, &pts, 15f32.to_radians(), diag * 0.03, 200);

        // Exterior = wall with the storey's points (almost) all on ONE side —
        // it lies on the footprint boundary; interior partitions have rooms on
        // both sides. LOCALISED to the wall's own footprint extent (along its
        // in-plane direction t): a short interior wall then sees both of its
        // adjacent rooms instead of the whole building, and a wall in a concave
        // notch still tests correctly. Points beyond the wall's span are ignored.
        let margin = diag * 0.02;
        let span = diag * 0.03; // slack beyond the wall's extent
        let (mut ne, mut ni) = (0, 0);
        for (n, d, gidx) in walls {
            // In-plane direction t = up × n, and this wall's extent along t.
            let t = unit(cross(up, n));
            let (mut tlo, mut thi) = (f32::MAX, f32::MIN);
            for &g in &gidx {
                let tp = dot(pts[g], t);
                tlo = tlo.min(tp);
                thi = thi.max(tp);
            }
            let (mut pos, mut neg) = (0u32, 0u32);
            for &g in storey_idx {
                let tp = dot(pts[g], t);
                if tp < tlo - span || tp > thi + span {
                    continue; // outside this wall's footprint span
                }
                let s = dot(n, pts[g]) + d;
                if s > margin {
                    pos += 1;
                } else if s < -margin {
                    neg += 1;
                }
            }
            let (mn, mx) = (pos.min(neg), pos.max(neg));
            let exterior = mx > 0 && (mn as f32) < 0.12 * mx as f32;
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
            ne + ni,
            raw_n = raw.len()
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
