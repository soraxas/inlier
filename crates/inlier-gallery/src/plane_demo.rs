use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use spatialrust_inlier::{
    PointCloudBuilder, StandardSchemas, point_cloud_to_data_matrix,
    plane::{estimate_plane_from_cloud, estimate_plane_magsac},
};

#[derive(Debug, Clone, PartialEq)]
pub enum ScoringMode {
    Msac,
    Magsac,
}

use crate::AppDemo;

pub struct PlanePlugin;

impl Plugin for PlanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlaneState>()
            .add_systems(OnEnter(AppDemo::Plane), on_enter)
            .add_systems(OnExit(AppDemo::Plane), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                plane_ui.run_if(in_state(AppDemo::Plane)),
            )
            .add_systems(
                Update,
                (plane_scene, apply_plane_visibility).chain().run_if(in_state(AppDemo::Plane)),
            );
    }
}

const MAX_PLANES: usize = 5;

// Point colors per plane index
const PT_COLORS: [[f32; 3]; MAX_PLANES] = [
    [0.05, 0.85, 0.15],
    [0.9,  0.75, 0.05],
    [0.05, 0.7,  0.9 ],
    [0.9,  0.3,  0.05],
    [0.7,  0.05, 0.9 ],
];

// Plane mesh colors per plane index (semi-transparent)
const MESH_COLORS: [[f32; 4]; MAX_PLANES] = [
    [0.2, 0.4, 0.9, 0.25],
    [0.9, 0.8, 0.1, 0.20],
    [0.1, 0.8, 0.8, 0.20],
    [0.9, 0.4, 0.1, 0.20],
    [0.7, 0.1, 0.8, 0.20],
];

#[derive(Resource)]
pub struct PlaneState {
    pub n_planes: usize,
    pub n_inliers: usize,
    pub noise_std: f32,
    pub n_outliers: usize,
    pub scoring: ScoringMode,
    pub threshold: f32,
    pub sigma_max: f32,
    pub min_pts_stop: usize,
    pub min_inlier_ratio: f32,
    pub seed: u32,
    pub auto_seed: bool,
    pub randomize_normal: bool,
    pub point_size: f32,
    // Plane 0 manual orientation (ignored when randomize_normal is on)
    pub true_nx: f32,
    pub true_ny: f32,
    pub true_nz: f32,
    pub true_d: f32,
    pub needs_run: bool,
    // (normal, d) per generated GT plane
    pub gt_planes: Vec<([f32; 3], f32)>,
    // (normal, d, inlier_count, iters) per detected plane
    pub est_planes: Vec<([f32; 3], f32, usize, usize)>,
    // visibility toggle per detected plane (indexed same as est_planes)
    pub plane_visible: Vec<bool>,
    pub error_msg: Option<String>,
}

impl Default for PlaneState {
    fn default() -> Self {
        Self {
            n_planes: 2,
            n_inliers: 300,
            noise_std: 0.03,
            n_outliers: 150,
            scoring: ScoringMode::Msac,
            threshold: 0.05,
            sigma_max: 0.2,
            min_pts_stop: 30,
            min_inlier_ratio: 0.05,
            seed: 42,
            auto_seed: true,
            randomize_normal: true,
            point_size: 0.04,
            true_nx: 0.0,
            true_ny: 1.0,
            true_nz: 0.0,
            true_d: 0.0,
            needs_run: true,
            gt_planes: vec![],
            est_planes: vec![],
            plane_visible: vec![],
            error_msg: None,
        }
    }
}

#[derive(Component)]
struct PlaneEntity;

/// Tags an inlier-point entity with which detected plane index it belongs to.
#[derive(Component)]
struct PlaneIndex(usize);

fn on_enter(mut state: ResMut<PlaneState>) {
    state.needs_run = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<PlaneEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn plane_ui(mut contexts: EguiContexts, mut state: ResMut<PlaneState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("plane_panel").default_size(250.0).show(ctx, |ui| {
        ui.heading("Multi-Plane Estimation");
        ui.separator();

        egui::CollapsingHeader::new("Generation").default_open(true).show(ui, |ui| {
            ui.add(egui::Slider::new(&mut state.n_planes, 1..=MAX_PLANES).text("Max planes"));
            ui.add(egui::Slider::new(&mut state.n_inliers, 50..=2000).text("Inliers / plane"));
            ui.add(egui::Slider::new(&mut state.noise_std, 0.001..=0.3_f32).text("Noise σ"));
            ui.add(egui::Slider::new(&mut state.n_outliers, 0..=1000).text("Outliers (total)"));
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut state.seed, 1..=9999_u32).text("Seed"));
                ui.checkbox(&mut state.auto_seed, "Auto");
            });
            ui.checkbox(&mut state.randomize_normal, "Randomize normals");
            if !state.randomize_normal {
                ui.label("Plane 0 normal:");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut state.true_nx).speed(0.01).prefix("x "));
                    ui.add(egui::DragValue::new(&mut state.true_ny).speed(0.01).prefix("y "));
                    ui.add(egui::DragValue::new(&mut state.true_nz).speed(0.01).prefix("z "));
                });
                ui.add(egui::Slider::new(&mut state.true_d, -3.0..=3.0_f32).text("d offset"));
            }
        });

        ui.separator();
        egui::CollapsingHeader::new("RANSAC").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Scoring:");
                ui.selectable_value(&mut state.scoring, ScoringMode::Msac, "MSAC");
                ui.selectable_value(&mut state.scoring, ScoringMode::Magsac, "MAGSAC++");
            });
            match state.scoring {
                ScoringMode::Msac => {
                    ui.horizontal(|ui| {
                        ui.add(egui::Slider::new(&mut state.threshold, 0.005..=0.3_f32).text("Threshold"));
                        if ui.button("Auto").clicked() {
                            state.threshold = (state.noise_std * 2.5).max(0.005);
                        }
                    });
                }
                ScoringMode::Magsac => {
                    ui.horizontal(|ui| {
                        ui.add(egui::Slider::new(&mut state.sigma_max, 0.01..=1.0_f32).text("σ max"));
                        if ui.button("Auto").clicked() {
                            state.sigma_max = (state.noise_std * 10.0).max(0.01);
                        }
                    });
                    ui.small("MAGSAC++ marginalises over noise — σ max is a loose upper bound");
                }
            }
            ui.add(egui::Slider::new(&mut state.min_pts_stop, 10..=500).text("Min pts to stop"));
            ui.add(
                egui::Slider::new(&mut state.min_inlier_ratio, 0.01..=0.5_f32)
                    .text("Min inlier ratio")
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
            );
            ui.small("Stop early if plane captures < ratio × total pts");
        });

        ui.separator();
        ui.add(egui::Slider::new(&mut state.point_size, 0.01..=0.15_f32).text("Point size"));

        ui.separator();
        if ui.button("Generate & Estimate").clicked() {
            if state.auto_seed {
                state.seed = lcg_next_u32(state.seed);
            }
            state.needs_run = true;
        }

        if let Some(ref msg) = state.error_msg.clone() {
            ui.colored_label(egui::Color32::RED, msg);
        }

        if !state.est_planes.is_empty() {
            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_BLUE, format!(
                "Detected {}/{} planes", state.est_planes.len(), state.n_planes
            ));
            for i in 0..state.est_planes.len() {
                let (n, d, inliers, iters) = state.est_planes[i];
                let [r, g, b] = PT_COLORS[i % MAX_PLANES];
                let color = egui::Color32::from_rgb(
                    (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                );
                ui.horizontal(|ui| {
                    if i < state.plane_visible.len() {
                        ui.checkbox(&mut state.plane_visible[i], "");
                    }
                    ui.colored_label(color, format!("Plane {}", i + 1));
                });
                ui.label(format!("  n=[{:.2},{:.2},{:.2}] d={:.3}", n[0], n[1], n[2], d));
                ui.label(format!("  inliers={} iters={}", inliers, iters));

                if let Some((gt_idx, err_deg)) = best_gt_match(n, &state.gt_planes) {
                    let ang_color = if err_deg < 2.0 { egui::Color32::GREEN } else { egui::Color32::YELLOW };
                    ui.colored_label(ang_color, format!(
                        "  → GT plane {} err={:.2}°", gt_idx + 1, err_deg
                    ));
                }
            }
        }

        ui.separator();
        for i in 0..state.n_planes.min(MAX_PLANES) {
            let [r, g, b] = PT_COLORS[i];
            let color = egui::Color32::from_rgb(
                (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
            );
            ui.colored_label(color, format!("■ Plane {}", i + 1));
        }
        ui.colored_label(egui::Color32::GRAY, "■ Unassigned outliers");
        ui.small("Faint mesh = GT  |  Solid mesh = detected");
    });
}

fn plane_scene(
    mut commands: Commands,
    mut state: ResMut<PlaneState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<PlaneEntity>>,
) {
    if !state.needs_run {
        return;
    }
    state.needs_run = false;

    for e in &existing {
        commands.entity(e).despawn();
    }

    // Build GT plane configs
    let gt_configs = build_gt_planes(
        state.n_planes,
        [state.true_nx, state.true_ny, state.true_nz],
        state.true_d,
        state.seed,
        state.randomize_normal,
    );

    // Apply back to state so UI reflects randomized normals
    state.gt_planes = gt_configs.iter().map(|&(n, d, _)| (n, d)).collect();
    if state.randomize_normal {
        if let Some(&(n, d, _)) = gt_configs.first() {
            state.true_nx = n[0]; state.true_ny = n[1]; state.true_nz = n[2];
            state.true_d = d;
        }
    }

    // Generate point cloud
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for &(normal, d, seed_i) in &gt_configs {
        let true_normal = normalize3(normal);
        let (inliers, _) = generate_plane_pts(
            true_normal, d, state.n_inliers, state.noise_std, 0, seed_i,
        );
        for p in inliers { let _ = builder.push_point(p); }
    }
    // Shared outliers
    let (_, outliers) = generate_plane_pts(
        [0.0, 1.0, 0.0], 0.0, 0, 0.0, state.n_outliers, state.seed.wrapping_add(0xdead),
    );
    for p in outliers { let _ = builder.push_point(p); }

    let full_cloud = match builder.build() {
        Ok(c) => c,
        Err(e) => { state.error_msg = Some(format!("Build error: {e}")); return; }
    };

    // Spawn GT plane meshes (very transparent)
    for (i, &(normal, d, _)) in gt_configs.iter().enumerate() {
        let [r, g, b, _] = MESH_COLORS[i % MAX_PLANES];
        spawn_plane(&mut commands, &mut meshes, &mut materials,
            normalize3(normal), d,
            Color::srgba(r * 0.6, g * 0.6, b * 0.6, 0.08),
            PlaneEntity);
    }

    // Sequential RANSAC — collect planes and their initial inliers.
    state.est_planes.clear();
    state.plane_visible.clear();
    state.error_msg = None;

    let total_pts = full_cloud.len();
    let min_inliers = ((total_pts as f32 * state.min_inlier_ratio) as usize).max(1);

    // (normal, d, iterations, inlier_positions)
    let mut detected: Vec<([f32;3], f32, usize, Vec<[f32;3]>)> = Vec::new();
    let mut remaining = full_cloud;
    let eps = state.threshold.max(state.sigma_max) * 10.0;

    for i in 0..state.n_planes {
        if remaining.len() < state.min_pts_stop { break; }
        let plane_result = match state.scoring {
            ScoringMode::Msac  => estimate_plane_from_cloud(&remaining, state.threshold as f64, None),
            ScoringMode::Magsac => estimate_plane_magsac(&remaining, Some(state.sigma_max as f64), None),
        };
        match plane_result {
            Ok(result) => {
                if result.inlier_cloud.len() < min_inliers { break; }
                let all_inlier_pos = cloud_xyz(&result.inlier_cloud);
                let (core_pos, rejected_pos) =
                    largest_connected_component_split(&all_inlier_pos, eps);
                if core_pos.len() < state.min_pts_stop {
                    let mut all_out = cloud_xyz(&result.outlier_cloud);
                    all_out.extend_from_slice(&rejected_pos);
                    let mut b = PointCloudBuilder::new(StandardSchemas::point_xyz());
                    for p in &all_out { let _ = b.push_point(*p); }
                    if let Ok(c) = b.build() { remaining = c; }
                    continue;
                }
                let mut all_out = cloud_xyz(&result.outlier_cloud);
                all_out.extend_from_slice(&rejected_pos);
                let mut b = PointCloudBuilder::new(StandardSchemas::point_xyz());
                for p in &all_out { let _ = b.push_point(*p); }
                remaining = b.build().unwrap_or(result.outlier_cloud);
                detected.push((result.normal, result.d, result.iterations, core_pos));
            }
            Err(e) => {
                state.error_msg = Some(format!("Plane {}: {e}", i + 1));
                break;
            }
        }
    }

    // ── Nearest-plane reassignment ──────────────────────────────────────────
    // Sequential RANSAC is greedy: planes found first steal points near the
    // intersection of two surfaces. Fix: pool all classified points and reassign
    // each to whichever detected plane it is geometrically closest to.
    if detected.len() > 1 {
        // Collect all classified points with their current plane index.
        let mut all_classified: Vec<([f32;3], usize)> = detected.iter().enumerate()
            .flat_map(|(pi, (_, _, _, pts))| pts.iter().map(move |&p| (p, pi)))
            .collect();

        // Reassign to nearest plane (point-to-plane distance |n·p + d|).
        for (p, best_plane) in &mut all_classified {
            let mut best_dist = f32::MAX;
            for (pi, &(n, d, _, _)) in detected.iter().enumerate() {
                let dist = (n[0]*p[0] + n[1]*p[1] + n[2]*p[2] + d).abs();
                if dist < best_dist { best_dist = dist; *best_plane = pi; }
            }
        }

        // Rebuild per-plane point lists from reassigned labels.
        let np = detected.len();
        let mut new_pts: Vec<Vec<[f32;3]>> = vec![Vec::new(); np];
        for (p, plane_idx) in all_classified {
            new_pts[plane_idx].push(p);
        }
        for (i, pts) in new_pts.into_iter().enumerate() {
            detected[i].3 = pts;
        }
    }

    // ── Spawn meshes from final reassigned inliers ──────────────────────────
    for (plane_idx, (normal, d, iters, core_pos)) in detected.iter().enumerate() {
        let plane_idx_u = state.est_planes.len();
        state.est_planes.push((*normal, *d, core_pos.len(), *iters));
        state.plane_visible.push(true);

        let [r, g, b] = PT_COLORS[plane_idx % MAX_PLANES];
        if !core_pos.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(make_cloud_mesh(core_pos, state.point_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(r, g, b),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                PlaneEntity,
                PlaneIndex(plane_idx_u),
            ));
        }
        let [r, g, b, a] = MESH_COLORS[plane_idx % MAX_PLANES];
        spawn_fitted_plane(&mut commands, &mut meshes, &mut materials,
            *normal, *d, core_pos,
            state.threshold, state.sigma_max,
            Color::srgba(r, g, b, a),
            PlaneEntity);
    }

    // Remaining unassigned points → gray
    let leftover = cloud_xyz(&remaining);
    if !leftover.is_empty() {
        commands.spawn((
            Mesh3d(meshes.add(make_cloud_mesh(&leftover, state.point_size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.45, 0.45, 0.45),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            PlaneEntity,
        ));
    }
}

fn apply_plane_visibility(
    state: Res<PlaneState>,
    mut query: Query<(&PlaneIndex, &mut Visibility)>,
) {
    for (idx, mut vis) in &mut query {
        let visible = state.plane_visible.get(idx.0).copied().unwrap_or(true);
        *vis = if visible { Visibility::Visible } else { Visibility::Hidden };
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build GT plane configs: (normal, d, per_plane_seed) for each plane.
fn build_gt_planes(
    n_planes: usize,
    plane0_normal: [f32; 3],
    plane0_d: f32,
    seed: u32,
    randomize: bool,
) -> Vec<([f32; 3], f32, u32)> {
    (0..n_planes).map(|i| {
        if i == 0 && !randomize {
            (plane0_normal, plane0_d, seed)
        } else {
            // Derive normal and d from a per-plane seed offset
            let plane_seed = seed.wrapping_add((i as u32).wrapping_mul(0x9e3779b9));
            let normal = random_unit_vec(plane_seed);
            // Space planes apart along their normals so they don't overlap
            let d = (i as f32) * 2.5;
            (normal, d, plane_seed)
        }
    }).collect()
}

/// Find the GT plane with smallest angular distance to `detected_normal`.
/// Returns (gt_index, angle_degrees).
fn best_gt_match(detected: [f32; 3], gt: &[([f32; 3], f32)]) -> Option<(usize, f32)> {
    gt.iter().enumerate().map(|(i, &(n, _))| {
        let dot = (n[0]*detected[0] + n[1]*detected[1] + n[2]*detected[2]).abs().clamp(0.0, 1.0);
        (i, dot.acos().to_degrees())
    }).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

/// Spawn an infinite 9×9 plane (used for GT ghost planes).
fn spawn_plane(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    normal: [f32; 3],
    d: f32,
    color: Color,
    marker: impl Bundle,
) {
    let n = Vec3::new(normal[0], normal[1], normal[2]).normalize();
    let rotation = if n.dot(Vec3::Y).abs() > 0.9999 {
        if n.y > 0.0 { Quat::IDENTITY } else { Quat::from_rotation_z(std::f32::consts::PI) }
    } else {
        Quat::from_rotation_arc(Vec3::Y, n)
    };
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(9.0, 9.0).build())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            unlit: true,
            ..default()
        })),
        Transform { translation: n * (-d), rotation, ..default() },
        marker,
    ));
}

/// Build an alpha-shape mesh (concave hull via Delaunay triangulation) for a plane patch.
///
/// Projects `inliers` onto the plane tangent frame, builds a 2-D Delaunay triangulation
/// with `spade`, then keeps only triangles whose circumradius ≤ alpha. Alpha is set to
/// `2 × median nearest-neighbour distance` (Jochem 2009 canonical rule).
pub fn make_alpha_shape_mesh(normal: [f32; 3], d: f32, inliers: &[[f32; 3]]) -> Option<Mesh> {
    use spade::{DelaunayTriangulation, Point2, Triangulation};

    if inliers.len() < 3 { return None; }

    let nv = Vec3::new(normal[0], normal[1], normal[2]).normalize();
    let t1 = if nv.dot(Vec3::Y).abs() > 0.9 { nv.cross(Vec3::Z).normalize() }
             else { nv.cross(Vec3::Y).normalize() };
    let t2 = nv.cross(t1).normalize();
    let origin = nv * (-d);

    let pts2d: Vec<[f64; 2]> = inliers.iter().map(|&p| {
        let rel = Vec3::from(p) - origin;
        [rel.dot(t1) as f64, rel.dot(t2) as f64]
    }).collect();

    let alpha = {
        let stride = (pts2d.len() / 300).max(1);
        let s: Vec<_> = pts2d.iter().step_by(stride).collect();
        let k = s.len();
        let mut nn: Vec<f64> = s.iter().enumerate().map(|(i, p)| {
            s.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, q)| ((p[0]-q[0]).powi(2) + (p[1]-q[1]).powi(2)).sqrt())
                .fold(f64::MAX, f64::min)
        }).collect();
        nn.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (nn[k / 2] * 2.0).max(1e-6)
    };

    let mut dt: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    for &[x, y] in &pts2d { let _ = dt.insert(Point2::new(x, y)); }

    let norm_arr = [nv.x, nv.y, nv.z];
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut norms: Vec<[f32; 3]> = Vec::new();
    let mut idx:   Vec<u32>      = Vec::new();

    for face in dt.inner_faces() {
        let [pa, pb, pc] = face.positions();
        let ax = pa.x; let ay = pa.y;
        let bx = pb.x; let by = pb.y;
        let cx = pc.x; let cy = pc.y;
        let d_val: f64 = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by));
        if d_val.abs() < 1e-12 { continue; }
        let ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / d_val;
        let uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / d_val;
        let r = ((ax-ux).powi(2) + (ay-uy).powi(2)).sqrt();
        if r > alpha { continue; }

        let base = verts.len() as u32;
        for [u, v] in [[ax,ay],[bx,by],[cx,cy]] as [[f64;2];3] {
            verts.push((origin + t1*(u as f32) + t2*(v as f32)).to_array());
            norms.push(norm_arr);
        }
        idx.extend([base, base+1, base+2, base, base+2, base+1]);
        if verts.len() > 0x00FF_FFFF { break; }
    }

    if verts.is_empty() { return None; }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_indices(Indices::U32(idx));
    Some(mesh)
}

fn spawn_fitted_plane(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    normal: [f32; 3],
    d: f32,
    inliers: &[[f32; 3]],
    _ransac_threshold: f32,
    _sigma_max: f32,
    color: Color,
    marker: impl Bundle,
) {
    let Some(mesh) = make_alpha_shape_mesh(normal, d, inliers) else { return };
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::default(),
        marker,
    ));
}

/// Split 3D inlier positions into (largest_component, rejected) using the
/// same grid union-find, working directly in 3D without needing a projection.
fn largest_connected_component_split(
    pts: &[[f32; 3]],
    eps: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let n = pts.len();
    if n <= 3 {
        return (pts.to_vec(), vec![]);
    }
    let eps = eps.max(1e-6);

    use std::collections::HashMap;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
        while p[x] != x { p[x] = p[p[x]]; x = p[x]; }
        x
    }
    fn union(p: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(p, a); let rb = find(p, b);
        if ra != rb { p[ra] = rb; }
    }

    let inv = 1.0 / eps;
    let cell_of = |i: usize| -> (i32, i32, i32) {
        ((pts[i][0] * inv).floor() as i32,
         (pts[i][1] * inv).floor() as i32,
         (pts[i][2] * inv).floor() as i32)
    };
    let mut grid: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();
    for i in 0..n { grid.entry(cell_of(i)).or_default().push(i); }

    let eps_sq = eps * eps;
    for i in 0..n {
        let (cx, cy, cz) = cell_of(i);
        for dx in -1i32..=1 { for dy in -1i32..=1 { for dz in -1i32..=1 {
            if let Some(nb) = grid.get(&(cx+dx, cy+dy, cz+dz)) {
                for &j in nb {
                    if j <= i { continue; }
                    let d = (pts[i][0]-pts[j][0]).powi(2)
                          + (pts[i][1]-pts[j][1]).powi(2)
                          + (pts[i][2]-pts[j][2]).powi(2);
                    if d <= eps_sq { union(&mut parent, i, j); }
                }
            }
        }}}
    }

    let roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for &r in &roots { *sizes.entry(r).or_insert(0) += 1; }
    let best = *sizes.iter().max_by_key(|&(_, &v)| v).unwrap().0;

    let mut core = Vec::new();
    let mut rejected = Vec::new();
    for i in 0..n {
        if roots[i] == best { core.push(pts[i]); }
        else { rejected.push(pts[i]); }
    }
    (core, rejected)
}

/// Extract the largest spatially-connected component from 2D projected inliers.
///

pub fn cloud_xyz(cloud: &spatialrust_inlier::PointCloud) -> Vec<[f32; 3]> {
    let Ok(dm) = point_cloud_to_data_matrix(cloud) else { return vec![] };
    (0..dm.n_points())
        .map(|i| [dm.get(i, 0) as f32, dm.get(i, 1) as f32, dm.get(i, 2) as f32])
        .collect()
}

/// Hard cap: a 32-bit index buffer supports at most 2^32/24 ≈ 178M verts per mesh.
/// We cap at 100k points to keep GPU memory and build time reasonable.
const MAX_MESH_POINTS: usize = 100_000;

pub fn make_cloud_mesh(positions: &[[f32; 3]], size: f32) -> Mesh {
    let h = size * 0.5;
    // Subsample deterministically if over the cap
    let positions: &[[f32; 3]] = if positions.len() > MAX_MESH_POINTS {
        &positions[..MAX_MESH_POINTS]
    } else {
        positions
    };
    let n = positions.len();
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(n * 24);
    let mut norms: Vec<[f32; 3]> = Vec::with_capacity(n * 24);
    let mut idx: Vec<u32> = Vec::with_capacity(n * 36);

    for (i, &[cx, cy, cz]) in positions.iter().enumerate() {
        let base = (i * 24) as u32;
        verts.extend([[cx+h,cy-h,cz-h],[cx+h,cy+h,cz-h],[cx+h,cy+h,cz+h],[cx+h,cy-h,cz+h]]);
        for _ in 0..4 { norms.push([1.,0.,0.]); }
        verts.extend([[cx-h,cy-h,cz+h],[cx-h,cy+h,cz+h],[cx-h,cy+h,cz-h],[cx-h,cy-h,cz-h]]);
        for _ in 0..4 { norms.push([-1.,0.,0.]); }
        verts.extend([[cx-h,cy+h,cz-h],[cx-h,cy+h,cz+h],[cx+h,cy+h,cz+h],[cx+h,cy+h,cz-h]]);
        for _ in 0..4 { norms.push([0.,1.,0.]); }
        verts.extend([[cx-h,cy-h,cz+h],[cx-h,cy-h,cz-h],[cx+h,cy-h,cz-h],[cx+h,cy-h,cz+h]]);
        for _ in 0..4 { norms.push([0.,-1.,0.]); }
        verts.extend([[cx-h,cy-h,cz+h],[cx+h,cy-h,cz+h],[cx+h,cy+h,cz+h],[cx-h,cy+h,cz+h]]);
        for _ in 0..4 { norms.push([0.,0.,1.]); }
        verts.extend([[cx+h,cy-h,cz-h],[cx-h,cy-h,cz-h],[cx-h,cy+h,cz-h],[cx+h,cy+h,cz-h]]);
        for _ in 0..4 { norms.push([0.,0.,-1.]); }
        for face in 0..6u32 {
            let b = base + face * 4;
            idx.extend([b, b+1, b+2, b+2, b+3, b]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_indices(Indices::U32(idx));
    mesh
}

fn generate_plane_pts(
    normal: [f32; 3],
    d: f32,
    n_in: usize,
    noise_std: f32,
    n_out: usize,
    seed: u32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut s: u64 = seed as u64;
    let mut rng = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let n = normal;
    let p0 = [-d*n[0], -d*n[1], -d*n[2]];
    let up = if n[2].abs() < 0.9 { [0.0_f32, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let t1 = normalize3(cross3(n, up));
    let t2 = cross3(n, t1);

    let inliers = (0..n_in).map(|_| {
        let u = rng() * 4.0;
        let v = rng() * 4.0;
        let noise = rng() * noise_std;
        [
            p0[0] + u*t1[0] + v*t2[0] + noise*n[0],
            p0[1] + u*t1[1] + v*t2[1] + noise*n[1],
            p0[2] + u*t1[2] + v*t2[2] + noise*n[2],
        ]
    }).collect();

    let outliers = (0..n_out)
        .map(|_| [rng()*4.0, rng()*4.0, rng()*4.0])
        .collect();

    (inliers, outliers)
}

fn lcg_next_u32(seed: u32) -> u32 {
    let s = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((s >> 33) as u32 % 9998) + 1
}

fn random_unit_vec(seed: u32) -> [f32; 3] {
    let mut s = seed as u64;
    let mut next = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    loop {
        let (x, y, z) = (next(), next(), next());
        let len = (x*x + y*y + z*z).sqrt();
        if len > 0.01 && len <= 1.0 {
            return [x/len, y/len, z/len];
        }
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt().max(1e-6);
    [v[0]/len, v[1]/len, v[2]/len]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
