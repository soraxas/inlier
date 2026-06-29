//! Gallery demo: end-to-end segmentation pipeline (region-grow → merge → grow).
//!
//! Shows the full Ling et al. 2024 pipeline in three interactive stages.
//! Each stage can be toggled on/off to see its effect:
//!
//!  1. **Segment** — region-growing + RANSAC; produces many small plane fragments.
//!  2. **Merge** — union-find co-planarity collapse; reduces fragment count.
//!  3. **Grow** — absorb leftover points into nearest matching plane.
//!
//! Colors are stable across stages (same palette index = same physical plane).

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use spatialrust_inlier::{
    auto_tune::auto_tune_settings,
    plane_ops::{GrowArgs, grow_planes, merge_planes},
    region_growing::{RansacMode, region_growing_ransac},
};

use crate::AppDemo;
use crate::merge_demo::point_cloud_cube_mesh;

pub struct SegmentationPlugin;

impl Plugin for SegmentationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SegmentationState>()
            .add_systems(OnEnter(AppDemo::Segmentation), on_enter)
            .add_systems(OnExit(AppDemo::Segmentation), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                segmentation_ui.run_if(in_state(AppDemo::Segmentation)),
            )
            .add_systems(Update, segmentation_scene.run_if(in_state(AppDemo::Segmentation)));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineView {
    AfterSegment,
    AfterMerge,
    AfterGrow,
}

#[derive(Resource)]
pub struct SegmentationState {
    pub view: PipelineView,
    pub show_unassigned: bool,
    pub point_size: f32,
    pub k: usize,
    pub angle_thresh_deg: f32,
    pub dist_thresh: f32,
    pub min_cluster_size: usize,
    pub merge_angle_deg: f32,
    pub merge_dist: f32,
    pub grow_dist: f32,
    pub auto_tune: bool,
    pub needs_update: bool,
    pub pts: Vec<[f32; 3]>,
    pub seg_planes: Vec<([f32; 3], f32, Vec<usize>)>,
    pub merge_planes_result: Vec<([f32; 3], f32, Vec<usize>)>,
    pub grow_planes_result: Vec<([f32; 3], f32, Vec<usize>)>,
    pub status: String,
}

impl Default for SegmentationState {
    fn default() -> Self {
        Self {
            view: PipelineView::AfterGrow,
            show_unassigned: true,
            point_size: 0.04,
            k: 20,
            angle_thresh_deg: 12.0,
            dist_thresh: 0.08,
            min_cluster_size: 40,
            merge_angle_deg: 6.0,
            merge_dist: 0.2,
            grow_dist: 0.15,
            auto_tune: false,
            needs_update: true,
            pts: Vec::new(),
            seg_planes: Vec::new(),
            merge_planes_result: Vec::new(),
            grow_planes_result: Vec::new(),
            status: "Ready.".into(),
        }
    }
}

#[derive(Component)]
struct SegmentationEntity;

fn on_enter(mut state: ResMut<SegmentationState>) {
    state.needs_update = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<SegmentationEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn segmentation_ui(mut contexts: EguiContexts, mut state: ResMut<SegmentationState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("seg_panel")
        .default_size(270.0)
        .show(ctx, |ui| {
            ui.heading("Segmentation Pipeline");
            ui.separator();

            ui.label("View stage:");
            ui.horizontal(|ui| {
                if ui.selectable_label(state.view == PipelineView::AfterSegment, "Segment").clicked() {
                    state.view = PipelineView::AfterSegment;
                    state.needs_update = true;
                }
                if ui.selectable_label(state.view == PipelineView::AfterMerge, "Merge").clicked() {
                    state.view = PipelineView::AfterMerge;
                    state.needs_update = true;
                }
                if ui.selectable_label(state.view == PipelineView::AfterGrow, "Grow").clicked() {
                    state.view = PipelineView::AfterGrow;
                    state.needs_update = true;
                }
            });

            ui.separator();
            let at = ui.checkbox(&mut state.auto_tune, "Auto-tune parameters").changed();
            if at { state.needs_update = true; }

            if !state.auto_tune {
                egui::CollapsingHeader::new("Segment parameters").default_open(true).show(ui, |ui| {
                    if ui.add(egui::Slider::new(&mut state.k, 5..=40).text("k neighbours")).changed() { state.needs_update = true; }
                    if ui.add(egui::Slider::new(&mut state.angle_thresh_deg, 3.0..=45.0_f32).text("Angle (°)")).changed() { state.needs_update = true; }
                    if ui.add(egui::Slider::new(&mut state.dist_thresh, 0.01..=0.5_f32).text("Dist thresh")).changed() { state.needs_update = true; }
                    if ui.add(egui::Slider::new(&mut state.min_cluster_size, 10..=200_usize).text("Min cluster")).changed() { state.needs_update = true; }
                });

                egui::CollapsingHeader::new("Merge parameters").default_open(true).show(ui, |ui| {
                    if ui.add(egui::Slider::new(&mut state.merge_angle_deg, 1.0..=20.0_f32).text("Angle (°)")).changed() { state.needs_update = true; }
                    if ui.add(egui::Slider::new(&mut state.merge_dist, 0.05..=1.0_f32).text("Dist")).changed() { state.needs_update = true; }
                });

                egui::CollapsingHeader::new("Grow parameters").default_open(true).show(ui, |ui| {
                    if ui.add(egui::Slider::new(&mut state.grow_dist, 0.02..=0.5_f32).text("Dist thresh")).changed() { state.needs_update = true; }
                });
            }

            ui.separator();
            let cu = ui.checkbox(&mut state.show_unassigned, "Show unassigned (gray)").changed();
            if cu { state.needs_update = true; }

            ui.separator();
            if ui.button("Re-run").clicked() { state.needs_update = true; }

            ui.separator();
            if !state.seg_planes.is_empty() {
                ui.colored_label(egui::Color32::LIGHT_GREEN, "Stats");
                ui.label(format!("After segment: {} planes", state.seg_planes.len()));
                ui.label(format!("After merge:   {} planes", state.merge_planes_result.len()));
                ui.label(format!("After grow:    {} planes", state.grow_planes_result.len()));
                let total_pts: usize = state.pts.len();
                let assigned: usize = state.grow_planes_result.iter().map(|(_, _, v)| v.len()).sum();
                if total_pts > 0 {
                    ui.label(format!("Assigned: {}/{} ({:.1}%)", assigned, total_pts, 100.0 * assigned as f32 / total_pts as f32));
                }
            }

            ui.separator();
            ui.small(&state.status);
        });
}

fn segmentation_scene(
    mut commands: Commands,
    mut state: ResMut<SegmentationState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<SegmentationEntity>>,
) {
    if !state.needs_update {
        return;
    }
    state.needs_update = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    let pts = synthetic_room();

    // Optionally apply auto-tune.
    let (angle_deg, dist, min_cluster, merge_angle_deg, merge_dist, grow_dist) = if state.auto_tune {
        let t = auto_tune_settings(&pts);
        state.status = t.description.clone();
        (t.angle_thresh, t.dist_thresh, t.min_cluster_size,
         t.merge_angle_thresh, t.merge_dist_thresh, t.grow_dist_thresh)
    } else {
        (state.angle_thresh_deg, state.dist_thresh, state.min_cluster_size,
         state.merge_angle_deg, state.merge_dist, state.grow_dist)
    };

    // Step 1: segment.
    let seg = region_growing_ransac(
        &pts, state.k, angle_deg.to_radians(), min_cluster, dist,
        RansacMode::Simple, 0.0, 500, 0.99,
    );

    // Step 2: merge.
    let merged = merge_planes(&seg, &pts, merge_angle_deg.to_radians(), merge_dist, min_cluster / 2);

    // Step 3: grow.
    let grow_args = GrowArgs {
        dist_thresh: grow_dist,
        use_normal: true,
        normal_cos_thresh: 35f32.to_radians().cos(),
        use_curvature: true,
        max_curvature: 0.08,
        use_connectivity: false,
    };
    let grown = grow_planes(&merged, &pts, &grow_args);

    if !state.auto_tune {
        state.status = format!(
            "Seg: {} → Merge: {} → Grow: {} planes",
            seg.len(), merged.len(), grown.len()
        );
    }

    let planes_to_show = match state.view {
        PipelineView::AfterSegment => &seg,
        PipelineView::AfterMerge => &merged,
        PipelineView::AfterGrow => &grown,
    };

    let palette: &[[f32; 3]] = &[
        [0.9, 0.2, 0.2], [0.2, 0.7, 0.9], [0.2, 0.9, 0.3],
        [0.9, 0.7, 0.1], [0.7, 0.2, 0.9], [0.9, 0.5, 0.1],
        [0.1, 0.9, 0.8], [0.8, 0.1, 0.5],
    ];

    let n_pts = pts.len();
    let mut assigned = vec![false; n_pts];

    for (pi, (_, _, inliers)) in planes_to_show.iter().enumerate() {
        let [r, g, b] = palette[pi % palette.len()];
        let pts3d: Vec<[f32; 3]> = inliers.iter().map(|&i| pts[i]).collect();
        for &i in inliers { if i < n_pts { assigned[i] = true; } }
        if pts3d.is_empty() { continue; }
        commands.spawn((
            Mesh3d(meshes.add(point_cloud_cube_mesh(&pts3d, state.point_size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(r, g, b),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            SegmentationEntity,
        ));
    }

    if state.show_unassigned {
        let unassigned: Vec<[f32; 3]> = pts.iter().enumerate()
            .filter(|(i, _)| !assigned[*i])
            .map(|(_, p)| *p)
            .collect();
        if !unassigned.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(point_cloud_cube_mesh(&unassigned, state.point_size * 0.7))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(0.45, 0.45, 0.45, 0.45),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                SegmentationEntity,
            ));
        }
    }

    state.pts = pts;
    state.seg_planes = seg;
    state.merge_planes_result = merged;
    state.grow_planes_result = grown;
}

/// Synthetic indoor room: floor + two walls + ceiling + a thin pillar.
fn synthetic_room() -> Vec<[f32; 3]> {
    let mut s: u64 = 0x1234_abcd;
    let mut rng = move || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut pts = Vec::new();
    let noise = 0.015f32;

    // Floor (y = 0).
    for _ in 0..800 {
        pts.push([rng() * 5.0, rng() * noise, rng() * 5.0]);
    }
    // Ceiling (y = 3).
    for _ in 0..600 {
        pts.push([rng() * 5.0, 3.0 + rng() * noise, rng() * 5.0]);
    }
    // Wall A (x = -2.5).
    for _ in 0..600 {
        pts.push([-2.5 + rng() * noise, rng() * 1.5, rng() * 5.0]);
    }
    // Wall B (x = +2.5).
    for _ in 0..600 {
        pts.push([2.5 + rng() * noise, rng() * 1.5, rng() * 5.0]);
    }
    // End wall (z = -2.5).
    for _ in 0..400 {
        pts.push([rng() * 5.0, rng() * 1.5, -2.5 + rng() * noise]);
    }
    // Sparse noise / furniture scatter.
    for _ in 0..200 {
        pts.push([rng() * 4.0, rng() * 3.0, rng() * 4.0]);
    }

    pts
}
