//! Gallery demo: per-point normal estimation and curvature heatmap.
//!
//! Shows the `pca_normal_and_curvature` + `spatial_grid::knn` pipeline on a
//! synthetic cloud that contains a flat floor, a tilted wall, and a curved arc.
//! Each point is colored by either its normal's Z-component (normal mode) or its
//! curvature value (heatmap mode).  Normal sticks are optionally drawn as thin
//! cylinders pointing in the normal direction.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use spatialrust_inlier::{
    normals::pca_normal_and_curvature,
    spatial_grid::{build_grid, estimate_cell_size, knn},
};

use crate::AppDemo;
use crate::plane_demo::make_cloud_mesh;

pub struct NormalsPlugin;

impl Plugin for NormalsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NormalsState>()
            .add_systems(OnEnter(AppDemo::Normals), on_enter)
            .add_systems(OnExit(AppDemo::Normals), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                normals_ui.run_if(in_state(AppDemo::Normals)),
            )
            .add_systems(Update, normals_scene.run_if(in_state(AppDemo::Normals)));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    NormalZ,
    Curvature,
}

#[derive(Resource)]
pub struct NormalsState {
    pub k: usize,
    pub point_size: f32,
    pub show_sticks: bool,
    pub color_mode: ColorMode,
    pub needs_update: bool,
    /// Computed: (position, unit_normal, curvature)
    pub results: Vec<([f32; 3], [f32; 3], f32)>,
    pub status: String,
}

impl Default for NormalsState {
    fn default() -> Self {
        Self {
            k: 20,
            point_size: 0.04,
            show_sticks: true,
            color_mode: ColorMode::NormalZ,
            needs_update: true,
            results: Vec::new(),
            status: "Ready.".into(),
        }
    }
}

#[derive(Component)]
struct NormalsEntity;

fn on_enter(mut state: ResMut<NormalsState>) {
    state.needs_update = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<NormalsEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn normals_ui(mut contexts: EguiContexts, mut state: ResMut<NormalsState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("normals_panel")
        .default_size(260.0)
        .show(ctx, |ui| {
            ui.heading("Normal Estimation");
            ui.separator();
            ui.label("Cloud: flat floor + tilted wall + curved arc (synthetic)");
            ui.separator();

            if ui
                .add(egui::Slider::new(&mut state.k, 5..=60).text("k neighbours"))
                .changed()
            {
                state.needs_update = true;
            }
            ui.add(egui::Slider::new(&mut state.point_size, 0.01..=0.15_f32).text("Point size"));

            ui.separator();
            ui.label("Color mode:");
            if ui
                .selectable_label(state.color_mode == ColorMode::NormalZ, "Normal Z (up = green)")
                .clicked()
            {
                state.color_mode = ColorMode::NormalZ;
            }
            if ui
                .selectable_label(state.color_mode == ColorMode::Curvature, "Curvature heatmap")
                .clicked()
            {
                state.color_mode = ColorMode::Curvature;
            }

            ui.separator();
            let changed = ui.checkbox(&mut state.show_sticks, "Show normal sticks").changed();
            if changed {
                state.needs_update = true;
            }

            ui.separator();
            if ui.button("Re-run").clicked() {
                state.needs_update = true;
            }
            ui.separator();
            ui.small(&state.status);

            if !state.results.is_empty() {
                let n_pts = state.results.len();
                let avg_curv: f32 =
                    state.results.iter().map(|(_, _, c)| c).sum::<f32>() / n_pts as f32;
                let max_curv = state
                    .results
                    .iter()
                    .map(|(_, _, c)| *c)
                    .fold(0.0f32, f32::max);
                ui.separator();
                ui.colored_label(egui::Color32::LIGHT_GREEN, "Stats");
                ui.label(format!("Points:     {n_pts}"));
                ui.label(format!("Avg curv:   {avg_curv:.4}"));
                ui.label(format!("Max curv:   {max_curv:.4}"));
            }
        });
}

fn normals_scene(
    mut commands: Commands,
    mut state: ResMut<NormalsState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<NormalsEntity>>,
) {
    if !state.needs_update {
        return;
    }
    state.needs_update = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    let pts = synthetic_cloud();
    let k = state.k;
    let cell_size = estimate_cell_size(&pts);
    let grid = build_grid(&pts, cell_size);

    let mut results: Vec<([f32; 3], [f32; 3], f32)> = Vec::with_capacity(pts.len());
    for i in 0..pts.len() {
        let neighbors = knn(&pts, i, k, cell_size, &grid);
        if let Some((normal, curvature)) = pca_normal_and_curvature(&pts, &neighbors) {
            results.push((pts[i], normal, curvature));
        }
    }

    state.status = format!("Estimated normals for {}/{} points (k={})", results.len(), pts.len(), k);

    let max_curv = results.iter().map(|(_, _, c)| *c).fold(1e-6f32, f32::max);

    // Build colored point cloud mesh.
    let point_positions: Vec<[f32; 3]> = results.iter().map(|(p, _, _)| *p).collect();
    let colors: Vec<[f32; 4]> = results
        .iter()
        .map(|(_, n, curv)| match state.color_mode {
            ColorMode::NormalZ => {
                let nz = n[2].abs();
                // nz≈1 (up/down) → green; nz≈0 (horizontal) → orange
                [1.0 - nz, 0.3 + 0.7 * nz, 0.1, 1.0]
            }
            ColorMode::Curvature => {
                let t = (curv / max_curv).clamp(0.0, 1.0);
                // cold (blue) → warm (red)
                [t, 0.2, 1.0 - t, 1.0]
            }
        })
        .collect();

    let mesh = make_colored_cloud_mesh(&point_positions, &colors, state.point_size);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            ..default()
        })),
        Transform::default(),
        NormalsEntity,
    ));

    if state.show_sticks {
        let stick_pts: Vec<[f32; 3]> = results
            .iter()
            .flat_map(|(p, n, _)| {
                let tip = [p[0] + n[0] * 0.12, p[1] + n[1] * 0.12, p[2] + n[2] * 0.12];
                [*p, tip]
            })
            .collect();

        let stick_mesh = make_cloud_mesh(&stick_pts, 0.01);
        commands.spawn((
            Mesh3d(meshes.add(stick_mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(1.0, 1.0, 0.2, 0.6),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            NormalsEntity,
        ));
    }

    state.results = results;
}

/// Synthetic cloud: flat XZ floor + tilted wall + curved arc.
fn synthetic_cloud() -> Vec<[f32; 3]> {
    let mut s: u64 = 0xdead_beef;
    let mut rng = move || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut pts = Vec::new();

    // Flat floor (y ≈ 0).
    for _ in 0..600 {
        pts.push([(rng() * 4.0), rng() * 0.02, rng() * 4.0]);
    }

    // Tilted wall (roughly normal [0.5, 0, 0.866], i.e. 30° from vertical).
    for _ in 0..300 {
        let u = rng() * 3.0;
        let v = rng() * 2.0;
        pts.push([2.0 + u * 0.866 + rng() * 0.02, v, -u * 0.5 + rng() * 0.02]);
    }

    // Curved arc (quarter-cylinder radius 1.5 centred at origin).
    for i in 0..200u32 {
        let theta = (i as f32 / 200.0) * std::f32::consts::FRAC_PI_2;
        let r = 1.5 + rng() * 0.02;
        let y = rng() * 1.5;
        pts.push([r * theta.cos(), y, r * theta.sin()]);
    }

    pts
}

/// Build a Bevy `Mesh` from point positions + per-point RGBA colors rendered as cubes.
fn make_colored_cloud_mesh(pts: &[[f32; 3]], colors: &[[f32; 4]], size: f32) -> Mesh {
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

    let n_pts = pts.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_pts * 8);
    let mut vertex_colors: Vec<[f32; 4]> = Vec::with_capacity(n_pts * 8);
    let mut indices: Vec<u32> = Vec::with_capacity(n_pts * 36);

    for (i, (p, col)) in pts.iter().zip(colors.iter()).enumerate() {
        let base = (i * 8) as u32;
        for v in &cube_verts {
            positions.push([p[0] + v[0], p[1] + v[1], p[2] + v[2]]);
            vertex_colors.push(*col);
        }
        for tri in &CUBE_TRIS {
            indices.push(base + tri[0]);
            indices.push(base + tri[1]);
            indices.push(base + tri[2]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
