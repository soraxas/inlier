//! Gallery demo: before/after merge of over-segmented planes.
//!
//! Generates a synthetic cloud with two large co-planar surfaces (same wall,
//! split by a gap) and shows how [`merge_planes`] collapses them into one.
//! A second patch at a different angle is left separate to confirm selectivity.
//!
//! UI: left panel shows segment count before/after; toggle raw segments / merged.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use spatialrust_inlier::{
    normals::normalize3,
    plane_ops::{GrowArgs, merge_planes, grow_planes},
    region_growing::{RansacMode, region_growing_ransac},
};

use crate::AppDemo;

pub struct MergePlugin;

impl Plugin for MergePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MergeState>()
            .add_systems(OnEnter(AppDemo::Merge), on_enter)
            .add_systems(OnExit(AppDemo::Merge), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                merge_ui.run_if(in_state(AppDemo::Merge)),
            )
            .add_systems(Update, merge_scene.run_if(in_state(AppDemo::Merge)));
    }
}

#[derive(Resource)]
pub struct MergeState {
    pub show_before: bool,
    pub show_after: bool,
    pub show_unassigned: bool,
    pub angle_thresh_deg: f32,
    pub dist_thresh: f32,
    pub min_pts: usize,
    pub needs_update: bool,
    pub before_planes: Vec<([f32; 3], f32, Vec<usize>)>,
    pub after_planes: Vec<([f32; 3], f32, Vec<usize>)>,
    pub all_pts: Vec<[f32; 3]>,
    pub status: String,
}

impl Default for MergeState {
    fn default() -> Self {
        Self {
            show_before: true,
            show_after: true,
            show_unassigned: true,
            angle_thresh_deg: 8.0,
            dist_thresh: 0.15,
            min_pts: 50,
            needs_update: true,
            before_planes: Vec::new(),
            after_planes: Vec::new(),
            all_pts: Vec::new(),
            status: "Ready.".into(),
        }
    }
}

#[derive(Component)]
struct MergeEntity;

fn on_enter(mut state: ResMut<MergeState>) {
    state.needs_update = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<MergeEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn merge_ui(mut contexts: EguiContexts, mut state: ResMut<MergeState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("merge_panel")
        .default_size(260.0)
        .show(ctx, |ui| {
            ui.heading("Merge Planes");
            ui.separator();
            ui.label("Synthetic cloud: two co-planar patches + one angled patch");
            ui.separator();

            let c1 = ui.checkbox(&mut state.show_before, "Show before (transparent)").changed();
            let c2 = ui.checkbox(&mut state.show_after, "Show after (solid)").changed();
            let c3 = ui.checkbox(&mut state.show_unassigned, "Show unassigned (gray)").changed();
            if c1 || c2 || c3 {
                state.needs_update = true;
            }

            ui.separator();
            if ui
                .add(
                    egui::Slider::new(&mut state.angle_thresh_deg, 1.0..=30.0_f32)
                        .text("Merge angle (°)"),
                )
                .changed()
            {
                state.needs_update = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.dist_thresh, 0.05..=1.0_f32).text("Merge dist"))
                .changed()
            {
                state.needs_update = true;
            }
            if ui.button("Re-run").clicked() {
                state.needs_update = true;
            }

            ui.separator();
            if !state.before_planes.is_empty() {
                ui.colored_label(egui::Color32::LIGHT_BLUE, "Stats");
                ui.label(format!("Segments before merge: {}", state.before_planes.len()));
                ui.label(format!("Segments after  merge: {}", state.after_planes.len()));
                let total: usize = state.after_planes.iter().map(|(_, _, v)| v.len()).sum();
                ui.label(format!("Total inliers after:   {total}"));
            }

            ui.separator();
            ui.small(&state.status);
        });
}

fn merge_scene(
    mut commands: Commands,
    mut state: ResMut<MergeState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<MergeEntity>>,
) {
    if !state.needs_update {
        return;
    }
    state.needs_update = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    let pts = synthetic_oversegmented_cloud();
    let planes = region_growing_ransac(
        &pts, 20, 12f32.to_radians(), 30, 0.08, RansacMode::Simple, 0.0, 500, 0.99,
    );

    let merged = merge_planes(
        &planes,
        &pts,
        state.angle_thresh_deg.to_radians(),
        state.dist_thresh,
        state.min_pts,
    );

    state.status = format!(
        "Before: {} segments → After: {} segments",
        planes.len(), merged.len()
    );

    // Track assigned indices.
    let n_pts = pts.len();
    let mut assigned = vec![false; n_pts];

    // Palette: distinct hues.
    let palette: &[[f32; 3]] = &[
        [0.9, 0.2, 0.2], [0.2, 0.7, 0.9], [0.2, 0.9, 0.3],
        [0.9, 0.7, 0.1], [0.7, 0.2, 0.9], [0.9, 0.5, 0.1],
    ];

    if state.show_before {
        for (pi, (_, _, inliers)) in planes.iter().enumerate() {
            let [r, g, b] = palette[pi % palette.len()];
            let pts3d: Vec<[f32; 3]> = inliers.iter().map(|&i| pts[i]).collect();
            if pts3d.is_empty() { continue; }
            commands.spawn((
                Mesh3d(meshes.add(point_cloud_cube_mesh(&pts3d, 0.04))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(r * 0.5, g * 0.5, b * 0.5, 0.3),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                MergeEntity,
            ));
        }
    }

    if state.show_after {
        for (pi, (_, _, inliers)) in merged.iter().enumerate() {
            let [r, g, b] = palette[pi % palette.len()];
            let pts3d: Vec<[f32; 3]> = inliers.iter().map(|&i| pts[i]).collect();
            for &i in inliers { assigned[i] = true; }
            if pts3d.is_empty() { continue; }
            commands.spawn((
                Mesh3d(meshes.add(point_cloud_cube_mesh(&pts3d, 0.04))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(r, g, b, 1.0),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                MergeEntity,
            ));
        }
    }

    if state.show_unassigned {
        let unassigned: Vec<[f32; 3]> = pts
            .iter()
            .enumerate()
            .filter(|(i, _)| !assigned[*i])
            .map(|(_, p)| *p)
            .collect();
        if !unassigned.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(point_cloud_cube_mesh(&unassigned, 0.03))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(0.5, 0.5, 0.5, 0.5),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                MergeEntity,
            ));
        }
    }

    state.before_planes = planes;
    state.after_planes = merged;
    state.all_pts = pts;
}

/// Synthetic scene: two co-planar patches (same XY plane, separated by a gap)
/// plus one angled patch.  The gap forces region-growing to produce 2 fragments
/// for the floor, which merge should collapse back to 1.
fn synthetic_oversegmented_cloud() -> Vec<[f32; 3]> {
    let mut s: u64 = 0xabad_cafe;
    let mut rng = move || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut pts = Vec::new();

    // Floor patch A (z ≈ 0, x in [-3.5, -0.5]).
    for _ in 0..500 {
        pts.push([(rng() * 1.5 - 2.0), rng() * 0.01, rng() * 3.0]);
    }
    // Floor patch B (z ≈ 0, x in [0.5, 3.5]) — same plane, gap in between.
    for _ in 0..500 {
        pts.push([(rng() * 1.5 + 2.0), rng() * 0.01, rng() * 3.0]);
    }
    // Angled wall (normal ≈ [0.707, 0.707, 0]).
    for _ in 0..400 {
        let u = rng() * 3.0;
        let v = rng() * 2.5;
        pts.push([u * 0.707 + rng() * 0.02, v, -u * 0.707 + rng() * 0.02]);
    }
    // Sparse noise.
    for _ in 0..100 {
        pts.push([rng() * 4.0, rng() * 3.0, rng() * 4.0]);
    }

    pts
}

pub fn point_cloud_cube_mesh(pts: &[[f32; 3]], size: f32) -> Mesh {
    let h = size * 0.5;
    let cube_verts: [[f32; 3]; 8] = [
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
    ];
    const CUBE_TRIS: [[u32; 3]; 12] = [
        [0,1,2],[0,2,3], [4,6,5],[4,7,6],
        [0,5,1],[0,4,5], [2,6,7],[2,7,3],
        [0,3,7],[0,7,4], [1,5,6],[1,6,2],
    ];

    let n = pts.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 8);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 36);

    for (i, p) in pts.iter().enumerate() {
        let base = (i * 8) as u32;
        for v in &cube_verts {
            positions.push([p[0] + v[0], p[1] + v[1], p[2] + v[2]]);
        }
        for tri in &CUBE_TRIS {
            indices.push(base + tri[0]);
            indices.push(base + tri[1]);
            indices.push(base + tri[2]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
