//! Full building pipeline with per-stage inspection dumps.
//!
//! Usage: inspect_building <input.ply> <outdir>
//!
//! Thin wrapper over [`spatialrust_inlier::reconstruct_building`] — the same code
//! the gallery's "Building" method runs. Writes four PolyFit `.vg` files into
//! <outdir>, coloured for `render6.py`:
//!   01_original — downsampled cloud (original coords; shows any tilt)
//!   02_aligned  — gravity/Manhattan-aligned coords (up → Z)
//!   03_storeys  — aligned, coloured per detected storey
//!   04_walls    — aligned, per-floor walls (red = exterior, blue = interior)

use std::io::Write;

use spatialrust_inlier::convert::point_cloud_to_data_matrix;
use spatialrust_inlier::io::read_point_cloud_file;
use spatialrust_inlier::{reconstruct_building, BuildingParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map(String::as_str).unwrap_or("SALON.ply");
    let outdir = args.get(2).map(String::as_str).unwrap_or("inspect-out");
    std::fs::create_dir_all(outdir)?;

    // Read the cloud and run the full pipeline (downsample happens inside).
    let cloud = read_point_cloud_file(input)?;
    let dm = point_cloud_to_data_matrix(&cloud)?;
    let raw: Vec<[f32; 3]> = (0..dm.n_points())
        .map(|i| [dm.get(i, 0) as f32, dm.get(i, 1) as f32, dm.get(i, 2) as f32])
        .collect();
    let scene = reconstruct_building(&raw, &BuildingParams::default());
    eprintln!(
        "{} points ({} raw) | up={:?} | {} storeys | {} walls",
        scene.points.len(),
        raw.len(),
        scene.up,
        scene.n_storeys,
        scene.walls.len()
    );

    let all: Vec<usize> = (0..scene.points.len()).collect();

    // Stage 1: original (downsampled) coords.
    write_vg(&format!("{outdir}/01_original.vg"), &scene.points, &[([0.6, 0.6, 0.6], &all)])?;

    // Stage 2: aligned coords.
    write_vg(&format!("{outdir}/02_aligned.vg"), &scene.aligned, &[([0.6, 0.6, 0.6], &all)])?;

    // Stage 3: aligned, coloured per storey.
    let palette = [[0.9, 0.4, 0.2], [0.2, 0.6, 0.9], [0.3, 0.85, 0.3], [0.8, 0.3, 0.9]];
    let mut per_storey_idx: Vec<Vec<usize>> = vec![Vec::new(); scene.n_storeys.max(1)];
    for (i, &s) in scene.storey_labels.iter().enumerate() {
        per_storey_idx[s].push(i);
    }
    let storey_groups: Vec<([f32; 3], Vec<usize>)> = per_storey_idx
        .iter()
        .enumerate()
        .map(|(si, idx)| (palette[si % palette.len()], idx.clone()))
        .collect();
    let sg: Vec<([f32; 3], &[usize])> =
        storey_groups.iter().map(|(c, v)| (*c, v.as_slice())).collect();
    write_vg(&format!("{outdir}/03_storeys.vg"), &scene.aligned, &sg)?;

    // Stage 4: aligned, walls coloured red = exterior / blue = interior.
    let (mut ne, mut ni) = (0, 0);
    let wall_groups: Vec<([f32; 3], Vec<usize>)> = scene
        .walls
        .iter()
        .map(|w| {
            let color = if w.is_exterior {
                ne += 1;
                [0.92, 0.22, 0.22]
            } else {
                ni += 1;
                [0.30, 0.60, 0.92]
            };
            (color, w.inlier_indices.clone())
        })
        .collect();
    let wg: Vec<([f32; 3], &[usize])> =
        wall_groups.iter().map(|(c, v)| (*c, v.as_slice())).collect();
    write_vg(&format!("{outdir}/04_walls.vg"), &scene.aligned, &wg)?;

    eprintln!("  {ne} exterior, {ni} interior walls — wrote 4 stages to {outdir}/");
    Ok(())
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
