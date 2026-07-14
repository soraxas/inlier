//! Gallery demo: absolute pose estimation (PnP / camera pose from 3D↔2D correspondences).
//!
//! Illustrates what "absolute pose" means geometrically:
//!   given a set of 3D world landmarks and their 2D image projections,
//!   we know where the camera is in the world.
//!
//! No RANSAC is performed at runtime — we use a known ground-truth pose and
//! show the geometry directly:
//!
//!  - Green cubes   : 3D world landmarks
//!  - Cyan box      : camera body at the estimated translation
//!  - Camera frustum: 4 rays from camera center to image-plane corners
//!                    + 4 edges forming the image-plane rectangle
//!  - Yellow rays   : projection rays for inlier observations (hit near 3D point)
//!  - Red rays      : projection rays for outlier observations (go somewhere random)

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

use crate::algo_config::{algo_combo_ui, randomize_button_ui, RansacAlgo};
use crate::merge_demo::point_cloud_cube_mesh;
use crate::AppDemo;

// ─── plugin ──────────────────────────────────────────────────────────────────

pub struct AbsolutePosePlugin;

impl Plugin for AbsolutePosePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AbsolutePoseState>()
            .add_systems(OnEnter(AppDemo::AbsolutePose), on_enter)
            .add_systems(OnExit(AppDemo::AbsolutePose), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                absolute_pose_ui.run_if(in_state(AppDemo::AbsolutePose)),
            )
            .add_systems(
                Update,
                absolute_pose_scene.run_if(in_state(AppDemo::AbsolutePose)),
            );
    }
}

// ─── state ───────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct AbsolutePoseState {
    pub n_inliers: usize,
    pub n_outliers: usize,
    pub rotation_y: f32,
    pub seed: u64,
    pub algo: RansacAlgo,
    pub needs_update: bool,
    pub n_world_points: usize,
    pub n_inliers_shown: usize,
    pub camera_pos: [f32; 3],
}

impl Default for AbsolutePoseState {
    fn default() -> Self {
        Self {
            n_inliers: 40,
            n_outliers: 10,
            rotation_y: 0.0,
            seed: 0xA85E_FACE_u64,
            algo: RansacAlgo::default(),
            needs_update: true,
            n_world_points: 0,
            n_inliers_shown: 0,
            camera_pos: [0.0; 3],
        }
    }
}

/// Marker component – every entity owned by this demo carries this tag.
#[derive(Component)]
pub struct AbsolutePoseEntity;

// ─── enter / exit ────────────────────────────────────────────────────────────

fn on_enter(mut state: ResMut<AbsolutePoseState>) {
    state.needs_update = true;
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<AbsolutePoseEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

// ─── UI ──────────────────────────────────────────────────────────────────────

fn absolute_pose_ui(mut contexts: EguiContexts, mut state: ResMut<AbsolutePoseState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("absolute_pose_panel")
        .default_size(270.0)
        .show(ctx, |ui| {
            ui.heading("Absolute Pose (PnP)");
            ui.separator();
            ui.label("3D world landmarks → 2D image projections → camera pose");
            ui.separator();

            if algo_combo_ui(ui, &mut state.algo) {
                state.needs_update = true;
            }
            ui.separator();

            if ui
                .add(egui::Slider::new(&mut state.n_inliers, 5..=200).text("Inliers"))
                .changed()
            {
                state.needs_update = true;
            }
            if ui
                .add(egui::Slider::new(&mut state.n_outliers, 0..=150).text("Outliers"))
                .changed()
            {
                state.needs_update = true;
            }
            if ui
                .add(
                    egui::Slider::new(&mut state.rotation_y, -0.5..=0.5_f32)
                        .text("Rotation Y (rad)")
                        .custom_formatter(|v, _| format!("{:.3}", v)),
                )
                .changed()
            {
                state.needs_update = true;
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Re-run").clicked() {
                    state.needs_update = true;
                }
                if randomize_button_ui(ui, &mut state.seed) {
                    state.needs_update = true;
                }
            });

            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Stats");
            ui.label(format!("World points:   {}", state.n_world_points));
            ui.label(format!("Inliers shown:  {}", state.n_inliers_shown));
            let [cx, cy, cz] = state.camera_pos;
            ui.label(format!("Camera pos:     ({:.3}, {:.3}, {:.3})", cx, cy, cz));

            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(80, 200, 80),
                "■ World landmarks (3D)",
            );
            ui.colored_label(egui::Color32::from_rgb(0, 200, 220), "■ Camera body");
            ui.colored_label(egui::Color32::from_rgb(200, 200, 60), "■ Inlier rays");
            ui.colored_label(egui::Color32::from_rgb(220, 60, 60), "■ Outlier rays");
            ui.colored_label(egui::Color32::WHITE, "■ Frustum / image plane");

            ui.separator();
            ui.small("Frustum: 4 rays from camera center to image corners");
            ui.small("Inlier rays pass through 3D point; outlier rays diverge");
        });
}

// ─── scene rebuild ───────────────────────────────────────────────────────────

fn absolute_pose_scene(
    mut commands: Commands,
    mut state: ResMut<AbsolutePoseState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<AbsolutePoseEntity>>,
) {
    if !state.needs_update {
        return;
    }
    state.needs_update = false;

    // Despawn previous geometry.
    for e in &existing {
        commands.entity(e).despawn();
    }

    // ── 1. Ground-truth pose ─────────────────────────────────────────────────
    // Rotation: Euler ZYX (roll=0.05, pitch=-0.18, yaw=0.12) + user rotation_y tweak.
    let base_rot = Quat::from_euler(EulerRot::ZYX, 0.05, -0.18, 0.12);
    let extra_rot = Quat::from_rotation_y(state.rotation_y);
    let cam_rot = extra_rot * base_rot;
    let cam_t = Vec3::new(0.18, -0.12, 0.45);

    state.camera_pos = cam_t.to_array();

    // ── 2. Generate 40 world points via LCG ─────────────────────────────────
    let all_world_pts = generate_world_points(state.n_inliers + state.n_outliers + 5, state.seed);
    state.n_world_points = all_world_pts.len();

    // Clamp sliders so we never ask for more points than generated.
    let n_in = state.n_inliers.min(all_world_pts.len());
    state.n_inliers_shown = n_in;

    // ── 3. Project inlier observations ──────────────────────────────────────
    // perspective: (X/Z, Y/Z) in camera space, then back-transform for display.
    let inlier_obs: Vec<Vec2> = all_world_pts[..n_in]
        .iter()
        .map(|&wp| {
            let cam_space = cam_rot.inverse() * (wp - cam_t);
            let z = cam_space.z.max(0.001);
            Vec2::new(cam_space.x / z, cam_space.y / z)
        })
        .collect();

    // ── 4. Generate outlier 2D observations via LCG ──────────────────────────
    let outlier_obs: Vec<Vec2> = generate_outlier_obs(state.n_outliers, 0xDEAD_BEEF_u64);

    // ── 5. Spawn world landmarks (green cubes) ───────────────────────────────
    let landmark_pts: Vec<[f32; 3]> = all_world_pts[..n_in].iter().map(|v| v.to_array()).collect();
    if !landmark_pts.is_empty() {
        commands.spawn((
            Mesh3d(meshes.add(point_cloud_cube_mesh(&landmark_pts, 0.07))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.85, 0.25),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            AbsolutePoseEntity,
        ));
    }

    // ── 6. Spawn camera body (small cyan box) ────────────────────────────────
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.12, 0.08, 0.06).mesh().build())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.85, 0.9),
            unlit: true,
            ..default()
        })),
        Transform {
            translation: cam_t,
            rotation: cam_rot,
            ..default()
        },
        AbsolutePoseEntity,
    ));

    // ── 7. Camera frustum ────────────────────────────────────────────────────
    // Image plane is at z=1 in camera space; half-extents ±0.6 horizontal, ±0.45 vertical.
    let hw = 0.6_f32;
    let hh = 0.45_f32;
    let image_depth = 1.0_f32;

    // Four image-plane corners in camera space.
    let corners_cam = [
        Vec3::new(-hw, -hh, image_depth),
        Vec3::new(hw, -hh, image_depth),
        Vec3::new(hw, hh, image_depth),
        Vec3::new(-hw, hh, image_depth),
    ];

    // Transform corners to world space.
    let corners_world: Vec<Vec3> = corners_cam.iter().map(|&c| cam_t + cam_rot * c).collect();

    // 4 rays: camera center → each corner.
    let ray_len = 1.0_f32; // the rays extend exactly to image_depth=1 then we extend further
                           // We'll draw each as a thin box mesh from cam_t to corner * (6/image_depth) to show depth.
    let frustum_extend = 5.0_f32; // extend rays 5 units past image plane
    let white_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.7),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });

    for &corner_w in &corners_world {
        let dir = (corner_w - cam_t).normalize();
        let end = cam_t + dir * (image_depth + frustum_extend);
        spawn_thin_line(
            &mut commands,
            &mut meshes,
            white_mat.clone(),
            cam_t,
            end,
            0.012,
            AbsolutePoseEntity,
        );
    }

    // 4 edges of the image-plane rectangle.
    for i in 0..4 {
        let a = corners_world[i];
        let b = corners_world[(i + 1) % 4];
        spawn_thin_line(
            &mut commands,
            &mut meshes,
            white_mat.clone(),
            a,
            b,
            0.012,
            AbsolutePoseEntity,
        );
    }

    // ── 8. Inlier projection rays (yellow) ───────────────────────────────────
    let yellow_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.92, 0.1),
        unlit: true,
        ..default()
    });

    for (obs, &wp) in inlier_obs.iter().zip(all_world_pts[..n_in].iter()) {
        // Direction in camera space: normalize(u, v, 1.0).
        let dir_cam = Vec3::new(obs.x, obs.y, 1.0).normalize();
        // World-space direction.
        let dir_world = cam_rot * dir_cam;
        // Extend the ray to reach roughly the z-depth of the world point.
        // The world point is at cam_t + dir_world * t for some t > 0.
        // t ≈ ‖wp - cam_t‖ along dir_world.
        let to_pt = wp - cam_t;
        let t = to_pt.dot(dir_world).max(0.2);
        let end = cam_t + dir_world * (t * 1.05); // slightly past the point
        spawn_thin_line(
            &mut commands,
            &mut meshes,
            yellow_mat.clone(),
            cam_t,
            end,
            0.009,
            AbsolutePoseEntity,
        );
    }

    // ── 9. Outlier projection rays (red) ─────────────────────────────────────
    let red_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.15, 0.15),
        unlit: true,
        ..default()
    });

    for obs in &outlier_obs {
        let dir_cam = Vec3::new(obs.x, obs.y, 1.0).normalize();
        let dir_world = cam_rot * dir_cam;
        let end = cam_t + dir_world * 6.0;
        spawn_thin_line(
            &mut commands,
            &mut meshes,
            red_mat.clone(),
            cam_t,
            end,
            0.009,
            AbsolutePoseEntity,
        );
    }
    let _ = ray_len; // suppress unused warning
}

// ─── geometry helpers ────────────────────────────────────────────────────────

/// Spawn a thin rectangular box as a line segment between `a` and `b`.
fn spawn_thin_line(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    material: Handle<StandardMaterial>,
    a: Vec3,
    b: Vec3,
    thickness: f32,
    marker: impl Bundle,
) {
    let mid = (a + b) * 0.5;
    let length = (b - a).length();
    if length < 1e-5 {
        return;
    }
    let dir = (b - a) / length;
    // Rotation from +Y (Bevy cuboid long axis) to dir.
    let rotation = Quat::from_rotation_arc(Vec3::Y, dir);

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(thickness, length, thickness).mesh().build())),
        MeshMaterial3d(material),
        Transform {
            translation: mid,
            rotation,
            ..default()
        },
        marker,
    ));
}

// ─── procedural data generation ──────────────────────────────────────────────

/// LCG step (Knuth multiplicative): produces a float in [0, 1).
#[inline]
fn lcg_next(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32) / (u32::MAX as f32 + 1.0)
}

/// Generate `n` 3D world points:
///   x ∈ [-1.5, 1.5], y ∈ [-1.2, 1.2], z ∈ [4.0, 7.5]
fn generate_world_points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let x = lcg_next(&mut s) * 3.0 - 1.5;
            let y = lcg_next(&mut s) * 2.4 - 1.2;
            let z = lcg_next(&mut s) * 3.5 + 4.0;
            Vec3::new(x, y, z)
        })
        .collect()
}

/// Generate `n` random 2D outlier observations in a plausible image region.
fn generate_outlier_obs(n: usize, seed: u64) -> Vec<Vec2> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let u = lcg_next(&mut s) * 2.4 - 1.2;
            let v = lcg_next(&mut s) * 1.8 - 0.9;
            Vec2::new(u, v)
        })
        .collect()
}

/// Build a line-segment mesh from pairs of points (used for debugging; kept for completeness).
#[allow(dead_code)]
fn line_strip_mesh(pts: &[Vec3]) -> Mesh {
    let positions: Vec<[f32; 3]> = pts.iter().map(|v| v.to_array()).collect();
    let indices: Vec<u32> = (0..positions.len() as u32).collect();
    let mut mesh = Mesh::new(PrimitiveTopology::LineStrip, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
