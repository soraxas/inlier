use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use spatialrust_inlier::{
    filter::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig},
    PointCloud, PointCloudBuilder, StandardSchemas,
};

use crate::plane_demo::{cloud_xyz, make_cloud_mesh};
use crate::AppDemo;

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelState>()
            .add_systems(OnEnter(AppDemo::Voxel), on_enter)
            .add_systems(OnExit(AppDemo::Voxel), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                voxel_ui.run_if(in_state(AppDemo::Voxel)),
            )
            .add_systems(Update, voxel_scene.run_if(in_state(AppDemo::Voxel)));
    }
}

#[derive(Resource)]
pub struct VoxelState {
    pub file_path: String,
    pub voxel_size: f32,
    pub point_size: f32,
    pub show_raw: bool,
    pub show_voxelized: bool,
    pub raw_pts: Option<Vec<[f32; 3]>>,
    pub vox_pts: Option<Vec<[f32; 3]>>,
    pub raw_count: usize,
    pub vox_count: usize,
    pub auto_size_hint: Option<f32>,
    pub needs_update: bool,
    pub needs_rerender: bool,
    pub status: String,
}

impl Default for VoxelState {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            voxel_size: 0.15,
            point_size: 0.04,
            show_raw: true,
            show_voxelized: true,
            raw_pts: None,
            vox_pts: None,
            raw_count: 0,
            vox_count: 0,
            auto_size_hint: None,
            needs_update: true,
            needs_rerender: false,
            status: "Click 'Synthetic' or enter a file path and Load.".into(),
        }
    }
}

#[derive(Component)]
struct VoxelEntity;

fn on_enter(mut state: ResMut<VoxelState>) {
    state.needs_update = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<VoxelEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn voxel_ui(mut contexts: EguiContexts, mut state: ResMut<VoxelState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("voxel_panel")
        .default_size(250.0)
        .show(ctx, |ui| {
            ui.heading("Voxel Downsample");
            ui.separator();

            ui.label("PLY / PCD / LAS file path:");
            ui.text_edit_singleline(&mut state.file_path);
            ui.horizontal(|ui| {
                if ui.button("Load").clicked() && !state.file_path.is_empty() {
                    state.needs_update = true;
                }
                if ui.button("Synthetic").clicked() {
                    state.file_path.clear();
                    state.needs_update = true;
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut state.voxel_size, 0.001..=2.0_f32)
                            .logarithmic(true)
                            .text("Voxel size (m)"),
                    )
                    .changed()
                {
                    state.needs_update = true;
                }
                if ui.button("Auto").clicked() {
                    if let Some(pts) = state.raw_pts.as_deref() {
                        let suggested = auto_voxel_size(pts);
                        state.voxel_size = suggested;
                        state.auto_size_hint = Some(suggested);
                        state.needs_update = true;
                    }
                }
            });
            if let Some(hint) = state.auto_size_hint {
                ui.small(format!("Auto-computed: {:.4} m  (mean NN dist × 3)", hint));
            }
            ui.add(egui::Slider::new(&mut state.point_size, 0.01..=0.15_f32).text("Point size"));

            ui.separator();
            let c1 = ui.checkbox(&mut state.show_raw, "Raw (gray)").changed();
            let c2 = ui
                .checkbox(&mut state.show_voxelized, "Voxelized (blue)")
                .changed();
            if c1 || c2 {
                state.needs_rerender = true;
            }

            if state.raw_count > 0 {
                ui.separator();
                ui.colored_label(egui::Color32::LIGHT_BLUE, "Stats");
                ui.label(format!("Raw:       {} pts", state.raw_count));
                ui.label(format!("Voxelized: {} pts", state.vox_count));
                let pct = 100.0 * state.vox_count as f32 / state.raw_count as f32;
                ui.label(format!("Retained:  {:.1}%", pct));
                ui.label(format!(
                    "Reduction: {:.1}×",
                    state.raw_count as f32 / state.vox_count.max(1) as f32
                ));
            }

            ui.separator();
            ui.small(&state.status);
        });
}

fn voxel_scene(
    mut commands: Commands,
    mut state: ResMut<VoxelState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<VoxelEntity>>,
) {
    if !state.needs_update && !state.needs_rerender {
        return;
    }

    if state.needs_update {
        state.needs_update = false;

        let raw = if state.file_path.is_empty() {
            state.status = "Using synthetic scene.".into();
            synthetic_cloud()
        } else {
            match spatialrust_inlier::io::read_point_cloud_file(&state.file_path) {
                Ok(c) => {
                    state.status = format!("Loaded {} pts", c.len());
                    c
                }
                Err(e) => {
                    state.status = format!("Error: {e}");
                    state.needs_rerender = false;
                    return;
                }
            }
        };

        let vox =
            match VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(state.voxel_size))
                .filter(&raw)
            {
                Ok(c) => c,
                Err(e) => {
                    state.status = format!("Voxelize error: {e}");
                    return;
                }
            };

        state.raw_count = raw.len();
        state.vox_count = vox.len();
        let raw_pts = cloud_xyz(&raw);
        state.auto_size_hint = None;
        state.raw_pts = Some(raw_pts);
        state.vox_pts = Some(cloud_xyz(&vox));
        state.status = format!(
            "{} → {} pts ({:.1}×)",
            state.raw_count,
            state.vox_count,
            state.raw_count as f32 / state.vox_count.max(1) as f32
        );
    }

    state.needs_rerender = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    let pt_sz = state.point_size;
    let vx_sz = state.voxel_size;

    if state.show_raw {
        if let Some(pts) = state.raw_pts.as_deref() {
            if !pts.is_empty() {
                commands.spawn((
                    Mesh3d(meshes.add(make_cloud_mesh(pts, pt_sz))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.55, 0.55, 0.55),
                        unlit: true,
                        ..default()
                    })),
                    Transform::default(),
                    VoxelEntity,
                ));
            }
        }
    }

    if state.show_voxelized {
        if let Some(pts) = state.vox_pts.as_deref() {
            if !pts.is_empty() {
                commands.spawn((
                    Mesh3d(meshes.add(make_cloud_mesh(pts, vx_sz))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgba(0.1, 0.4, 0.9, 0.85),
                        alpha_mode: AlphaMode::Blend,
                        unlit: true,
                        ..default()
                    })),
                    Transform::default(),
                    VoxelEntity,
                ));
            }
        }
    }
}

/// Estimate a good voxel size from the point cloud density.
///
/// Subsamples up to 300 points, computes mean nearest-neighbor distance,
/// and returns that × 3. Falls back to the cube-root of bounding-box volume
/// per point when there are fewer than 20 points.
pub fn auto_voxel_size(pts: &[[f32; 3]]) -> f32 {
    const SAMPLE: usize = 300;
    const FACTOR: f32 = 3.0;

    if pts.len() < 2 {
        return 0.1;
    }

    // Bounding-box fallback for very sparse clouds.
    if pts.len() < 20 {
        let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in pts {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        let vol = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]);
        return (vol / pts.len() as f32).cbrt().max(1e-4);
    }

    // Deterministic subsample via stride.
    let stride = (pts.len() / SAMPLE).max(1);
    let sample: Vec<[f32; 3]> = pts.iter().step_by(stride).cloned().collect();
    let k = sample.len();

    // O(k²) nearest-neighbor distances on the subsample.
    let mut sum = 0.0f32;
    for i in 0..k {
        let mut best = f32::MAX;
        for j in 0..k {
            if i == j {
                continue;
            }
            let dx = sample[i][0] - sample[j][0];
            let dy = sample[i][1] - sample[j][1];
            let dz = sample[i][2] - sample[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < best {
                best = d;
            }
        }
        sum += best;
    }
    let mean_nn = sum / k as f32;
    (mean_nn * FACTOR).max(1e-4)
}

fn synthetic_cloud() -> PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    let mut s: u64 = 0xcafe_babe;
    let mut rng = move || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };

    for _ in 0..2000 {
        let _ = builder.push_point([(rng() - 0.5) * 10., rng() * 0.02, (rng() - 0.5) * 10.]);
    }
    for _ in 0..400 {
        let _ = builder.push_point([2. + rng() * 1.2, rng() * 1.5, 1. + rng() * 1.2]);
    }
    for _ in 0..300 {
        let _ = builder.push_point([-3. + rng(), rng() * 1., -2. + rng()]);
    }
    for _ in 0..200 {
        let _ = builder.push_point([(rng() - 0.5) * 8., rng() * 3. + 1.5, (rng() - 0.5) * 8.]);
    }
    builder.build().unwrap()
}
