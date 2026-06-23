mod plane_demo;
mod voxel_demo;
mod voxel_plane_demo;

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppDemo {
    #[default]
    Plane,
    Voxel,
    VoxelPlane,
}

#[derive(Component)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub focus: Vec3,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "SpatialRust Viewer".into(),
                resolution: (1280u32, 800u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .init_state::<AppDemo>()
        .add_plugins(plane_demo::PlanePlugin)
        .add_plugins(voxel_demo::VoxelPlugin)
        .add_plugins(voxel_plane_demo::VoxelPlanePlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(EguiPrimaryContextPass, demo_switcher_ui)
        .add_systems(Update, orbit_camera_system)
        .run();
}

fn setup_scene(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4.0, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera { yaw: 0.0, pitch: -0.4, radius: 10.0, focus: Vec3::ZERO },
    ));
    commands.spawn((
        DirectionalLight { illuminance: 5000.0, ..default() },
        Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn demo_switcher_ui(
    mut contexts: EguiContexts,
    state: Res<State<AppDemo>>,
    mut next: ResMut<NextState<AppDemo>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::top("top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Demo:");
            if ui.selectable_label(state.get() == &AppDemo::Plane, "Plane Estimation").clicked() {
                next.set(AppDemo::Plane);
            }
            if ui.selectable_label(state.get() == &AppDemo::Voxel, "Voxel Downsample").clicked() {
                next.set(AppDemo::Voxel);
            }
            if ui.selectable_label(state.get() == &AppDemo::VoxelPlane, "Voxel-Normal Planes").clicked() {
                next.set(AppDemo::VoxelPlane);
            }
        });
    });
}

fn orbit_camera_system(
    mut contexts: EguiContexts,
    mouse: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let egui_wants = contexts
        .ctx_mut()
        .map(|ctx| ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui())
        .unwrap_or(false);

    let Ok((mut transform, mut cam)) = query.single_mut() else { return };

    if !egui_wants {
        if mouse_buttons.pressed(MouseButton::Left) {
            cam.yaw -= mouse.delta.x * 0.005;
            cam.pitch = (cam.pitch - mouse.delta.y * 0.005).clamp(-1.45, 1.45);
        }
        cam.radius = (cam.radius - scroll.delta.y * 0.5).clamp(0.5, 150.0);
    }

    let x = cam.radius * cam.pitch.cos() * cam.yaw.sin();
    let y = cam.radius * cam.pitch.sin();
    let z = cam.radius * cam.pitch.cos() * cam.yaw.cos();
    transform.translation = cam.focus + Vec3::new(x, y, z);
    transform.look_at(cam.focus, Vec3::Y);
}
