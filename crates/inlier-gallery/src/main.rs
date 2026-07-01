mod absolute_pose_demo;
mod algo_config;
mod essential_matrix_demo;
mod fundamental_matrix_demo;
mod line_fitting_demo;
mod merge_demo;
mod normals_demo;
mod plane_demo;
mod rigid_transform_demo;
mod segmentation_demo;
mod voxel_demo;
mod voxel_plane_demo;

// Threads build (wasm): re-export wasm-bindgen-rayon's pool initializer so JS
// can `await initThreadPool(n)` before the app runs. No-op on other builds.
#[cfg(all(target_arch = "wasm32", feature = "threads"))]
pub use wasm_bindgen_rayon::init_thread_pool;

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppDemo {
    #[default]
    Plane,
    Voxel,
    VoxelPlane,
    Normals,
    Merge,
    Segmentation,
    LineFit,
    AbsolutePose,
    EssentialMatrix,
    FundamentalMatrix,
    RigidTransform,
}

impl AppDemo {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Plane          => "Plane Estimation",
            Self::Voxel          => "Voxel Downsample",
            Self::VoxelPlane     => "Voxel-Normal Planes",
            Self::Normals        => "Normal Estimation",
            Self::Merge          => "Merge Demo",
            Self::Segmentation   => "Segmentation Pipeline",
            Self::LineFit        => "Line Fitting",
            Self::AbsolutePose   => "Absolute Pose",
            Self::EssentialMatrix   => "Essential Matrix",
            Self::FundamentalMatrix => "Fundamental Matrix",
            Self::RigidTransform => "Rigid Transform",
        }
    }

    pub fn all() -> &'static [AppDemo] {
        &[
            Self::Plane, Self::Voxel, Self::VoxelPlane,
            Self::Normals, Self::Merge, Self::Segmentation,
            Self::LineFit, Self::AbsolutePose,
            Self::EssentialMatrix, Self::FundamentalMatrix, Self::RigidTransform,
        ]
    }
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
                // On wasm, render into the <canvas id="bevy"> in index.html and
                // track its parent's size. Both fields are ignored on native.
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .init_state::<AppDemo>()
        .add_plugins(plane_demo::PlanePlugin)
        .add_plugins(voxel_demo::VoxelPlugin)
        .add_plugins(voxel_plane_demo::VoxelPlanePlugin)
        .add_plugins(normals_demo::NormalsPlugin)
        .add_plugins(merge_demo::MergePlugin)
        .add_plugins(segmentation_demo::SegmentationPlugin)
        .add_plugins(line_fitting_demo::LineFitPlugin)
        .add_plugins(absolute_pose_demo::AbsolutePosePlugin)
        .add_plugins(essential_matrix_demo::EssentialMatrixPlugin)
        .add_plugins(fundamental_matrix_demo::FundamentalMatrixPlugin)
        .add_plugins(rigid_transform_demo::RigidTransformPlugin)
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
            let current = state.get();
            egui::ComboBox::from_id_salt("demo_selector")
                .selected_text(current.label())
                .width(220.0)
                .show_ui(ui, |ui| {
                    for demo in AppDemo::all() {
                        let selected = current == demo;
                        if ui.selectable_label(selected, demo.label()).clicked() {
                            next.set(demo.clone());
                        }
                    }
                });
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
