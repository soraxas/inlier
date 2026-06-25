//! # Voxel-Normal Planes demo
//!
//! Interactive implementation of the Ling et al. ISPRS 2024 region-growing +
//! per-cluster RANSAC plane segmentation pipeline, extended with post-processing
//! (merge, grow) and a dollhouse cutaway visualisation.
//!
//! ---
//!
//! ## Full pipeline overview
//!
//! ```text
//! Load PLY / generate synthetic
//!        │
//!        ▼
//! [Step 1] Per-point PCA normal + curvature
//!        │  k-NN via 27-cell approximate grid search
//!        │  Covariance → smallest eigenvector (surface normal)
//!        │  Curvature = λ_min / trace
//!        │
//!        ▼
//! [Step 2] Region growing (Ling 2024 §3.1)
//!        │  Sort all points by curvature ascending (flattest = best seed)
//!        │  BFS grow: accept neighbour if |n_cur · n_nb| > cos(angle_thresh)
//!        │  Discard clusters < min_cluster_size
//!        │
//!        ▼
//! [Step 3] Per-cluster RANSAC + aggregation sweep
//!        │  Fit plane to each cluster (Simple / MSAC / MAGSAC++)
//!        │  Accept inliers within dist_thresh
//!        │  Aggregation: sweep entire cloud, absorb unassigned points that
//!        │    pass both distance AND normal angle tests
//!        │
//!        ▼  raw_planes  (normal, d, inlier_indices)
//!
//! [Optional] Merge
//!        │  Union-find on plane pairs where:
//!        │    |n_i · n_j| > cos(merge_angle_thresh)  AND
//!        │    |d_i - d_j| < merge_dist_thresh
//!        │  Refit merged groups via PCA; drop groups < merge_min_pts
//!        │
//!        ▼  merged_planes
//!
//! [Optional] Grow  (iterates up to grow_max_iters times)
//!        │  For each unassigned point, find nearest plane within grow_dist_thresh
//!        │  Three optional furniture filters:
//!        │    • Normal:       |pt_normal · plane_normal| > cos(grow_normal_angle)
//!        │    • Curvature:    pt_curvature < grow_max_curvature
//!        │    • Connectivity: ≥1 of point's 27-cell grid neighbours is an existing inlier
//!        │  Refit each expanded plane via PCA; repeat until stable
//!        │
//!        ▼  merged_planes  (updated in-place)
//!
//! [Visualisation] Spawn per-plane point cloud + alpha-shape mesh entities
//!        │  Normal canonicalization: orient each plane normal inward (toward
//!        │    majority of cloud points, ignoring the near-plane margin 3×dist_thresh)
//!        │  Exterior classification: outward_frac = fraction of far points on
//!        │    air side; plane is exterior if outward_frac < exterior_thresh (0.05)
//!        │
//!        ▼
//! [Dollhouse] Per-frame hiding of camera-facing planes (optional)
//!        │  For each exterior (or all) plane:
//!        │    dot = canonical_inward_normal · (cam_pos − centroid).normalise()
//!        │    if dot < −cos(dollhouse_angle) → Visibility::Hidden
//! ```
//!
//! ---
//!
//! ## Key data structures
//!
//! ### Plane representation
//! Every plane is stored as `([f32; 3], f32, Vec<usize>)` = `(normal, d, inlier_indices)`.
//! The plane equation is `normal · p + d = 0` (unit normal).  Inlier indices index into
//! the `all_pts_cache` flat array, not into the original file.
//!
//! ### `detected: Vec<([f32;3], f32, usize, [f32;3], bool)>`
//! Derived from the current planes at spawn time.  Each entry is
//! `(canonical_normal, d, count, centroid, is_exterior)`.
//! `canonical_normal` is oriented inward (toward the majority of cloud points),
//! which makes the dollhouse dot-product test unambiguous.
//!
//! ### Spatial grid (`HashMap<(i32,i32,i32), Vec<usize>>`)
//! Built by `build_grid`.  Cell size is chosen by `estimate_cell_size` so that each cell
//! contains ~8 points on average, meaning the 3×3×3 = 27-cell neighbourhood of any query
//! point reliably covers k = 20 neighbours.
//! Formula: `cell_size = (bbox_volume / n)^(1/3) × 2.0`.
//!
//! ---
//!
//! ## Algorithm parameters (UI sliders)
//!
//! | Parameter | Default | Effect |
//! |-----------|---------|--------|
//! | `k_neighbors` | 20 | k-NN size for normal/curvature estimation |
//! | `angle_thresh` | 10° | Max normal deviation to grow into a neighbour |
//! | `min_cluster_size` | 30 | Minimum region-growing cluster to run RANSAC on |
//! | `dist_thresh` | 0.08 m | RANSAC inlier distance band |
//! | `merge_angle_thresh` | 5° | Max normal angle to consider two planes co-planar |
//! | `merge_dist_thresh` | 0.15 m | Max plane-offset difference to consider same wall |
//! | `merge_min_pts` | 5000 | Drop merged plane if fewer points |
//! | `grow_dist_thresh` | 0.20 m | Max point-to-plane distance during grow step |
//! | `grow_max_iters` | 3 | Grow → refit iterations (stops early if stable) |
//! | `grow_normal_angle` | 30° | Furniture filter: point normal vs plane normal |
//! | `grow_max_curvature` | 0.05 | Furniture filter: skip high-curvature points |
//! | `dollhouse_angle` | 60° | Half-angle of "facing camera" cone |
//! | `dollhouse_exterior_thresh` | 0.05 | Outward fraction below which a plane is exterior |
//!
//! ---
//!
//! ## RANSAC mode comparison
//!
//! | Mode | Algorithm | Notes |
//! |------|-----------|-------|
//! | **Simple** | 3-point RANSAC + LS refine | Built-in, no deps, fast |
//! | **MSAC** | MSAC with IRLS local optimisation | Softer inlier weighting, more accurate |
//! | **MAGSAC++** | σ-consensus, threshold-free | Best for mixed-noise scans; slower |
//!
//! ---
//!
//! ## Normal canonicalization and exterior detection
//!
//! PCA normals have an arbitrary sign (pointing either side of the surface).  At spawn
//! time, `spawn_planes` canonicalizes each normal to point **inward** (toward the bulk of
//! the scan) by counting how many cloud points lie clearly on each side of the plane
//! (excluding the near-plane band `|n·p + d| < 3 × dist_thresh` which straddles both
//! sides due to noise).  The majority side wins.
//!
//! Exterior classification uses the same counts: if the minority (outward / air) side
//! contains fewer than `exterior_thresh` (default 5%) of the far points, the plane has
//! almost nothing beyond it and is classified as an exterior surface.  This threshold
//! must exclude near-plane points; without the margin, a large wall's own inliers
//! (~16% of cloud) would inflate the outward count above 5% even for a genuine exterior.
//!
//! ---
//!
//! ## Auto-tune
//!
//! `auto_tune_settings` samples ~2000 representative points, computes 20-NN for each,
//! and derives:
//! - `noise_sigma` = median of per-point plane residuals (sensor measurement noise)
//! - `point_spacing` = median nearest-neighbour distance (surface point density)
//!
//! From these it derives all algorithm thresholds:
//! ```text
//! dist_thresh        = noise_sigma × 4       (clamped [0.01, 0.5])
//! angle_thresh       = 15° + noise_sigma×200° (clamped [5°, 45°])
//! min_cluster_size   = 0.5 m² × density       (clamped [20, 2000])
//! merge_angle_thresh = angle_thresh × 0.4     (clamped [2°, 10°])
//! merge_dist_thresh  = dist_thresh × 3        (clamped [0.05, 1.0])
//! grow_dist_thresh   = dist_thresh × 2        (clamped [0.02, 1.0])
//! ```
//!
//! ---
//!
//! ## Grow step: furniture filtering rationale
//!
//! A plain distance-only grow absorbs furniture that happens to be close to a wall.
//! Three independent filters (all on by default, each individually toggleable) address
//! different failure modes:
//!
//! - **Normal filter**: A table top near a vertical wall has a horizontal normal (~90° off).
//!   Threshold 30° (cos ≈ 0.87) rejects this cleanly.  Does not help when furniture is
//!   oriented parallel to the wall (e.g. a bookshelf flush against a wall).
//!
//! - **Curvature filter**: Furniture edges and legs have high local curvature.  Threshold
//!   0.05 keeps only locally planar points.  Can over-reject wall corners.
//!
//! - **Connectivity filter**: Only absorb a point if one of its 27-cell grid neighbours is
//!   already an inlier of the target plane.  Physically disconnected fragments (scan gap,
//!   occluded section) cannot bridge across air.  **Limitation**: furniture that physically
//!   touches a wall (bookcases, radiators) passes this filter.  It is most useful for
//!   completely isolated objects.
//!
//! Iterating grow (default 3 iterations) lets wall fragments that were reachable only via
//! an intermediate neighbour get absorbed in later passes, while the plane is continually
//! refit to its expanded inlier set so the normal and offset stay accurate.
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use std::collections::{HashMap, VecDeque};

use crate::AppDemo;
use crate::plane_demo::{make_cloud_mesh, make_alpha_shape_mesh, cloud_xyz};
use crate::OrbitCamera;
use spatialrust_inlier::{fit_plane_msac, fit_plane_magsac_raw, MetasacSettings, io::read_point_cloud_file};

pub struct VoxelPlanePlugin;

impl Plugin for VoxelPlanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelPlaneState>()
            .add_systems(OnEnter(AppDemo::VoxelPlane), on_enter)
            .add_systems(OnExit(AppDemo::VoxelPlane), on_exit)
            .add_systems(
                EguiPrimaryContextPass,
                voxel_plane_ui.run_if(in_state(AppDemo::VoxelPlane)),
            )
            .add_systems(Update, voxel_plane_scene.run_if(in_state(AppDemo::VoxelPlane)))
            .add_systems(Update, dollhouse_system.run_if(in_state(AppDemo::VoxelPlane)));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointSource {
    Synthetic,
    File,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RansacMode {
    /// Pure Rust 3-pt RANSAC (always available, no external deps).
    Simple,
    /// inlier MSAC + IRLS local optimisation.
    Msac,
    /// MAGSAC++ σ-consensus (threshold-free, marginalises noise scale).
    Magsac,
}

const MAX_PLANES: usize = 8;

const PT_COLORS: [[f32; 3]; MAX_PLANES] = [
    [0.05, 0.85, 0.15],
    [0.9,  0.75, 0.05],
    [0.05, 0.7,  0.9 ],
    [0.9,  0.3,  0.05],
    [0.7,  0.05, 0.9 ],
    [0.9,  0.05, 0.4 ],
    [0.05, 0.5,  0.3 ],
    [0.5,  0.5,  0.05],
];

const MESH_COLORS: [[f32; 4]; MAX_PLANES] = [
    [0.2, 0.9, 0.3, 0.22],
    [0.9, 0.8, 0.1, 0.20],
    [0.1, 0.8, 0.9, 0.20],
    [0.9, 0.4, 0.1, 0.20],
    [0.7, 0.1, 0.8, 0.20],
    [0.9, 0.1, 0.4, 0.20],
    [0.1, 0.6, 0.4, 0.20],
    [0.6, 0.6, 0.1, 0.20],
];

#[derive(Resource)]
pub struct VoxelPlaneState {
    pub n_planes: usize,
    pub n_inliers: usize,
    pub noise_std: f32,
    pub n_outliers: usize,
    pub seed: u32,
    pub auto_seed: bool,
    pub point_size: f32,
    // Step 1
    pub k_neighbors: usize,
    // Step 2
    pub angle_thresh: f32,       // degrees in UI, radians internally
    pub min_cluster_size: usize,
    // Step 3
    pub dist_thresh: f32,
    pub ransac_mode: RansacMode,
    /// sigma_max for MAGSAC++ (multiples of dist_thresh; UI shows the multiplier)
    pub sigma_factor: f32,
    /// MetaSAC max iterations (MSAC + MAGSAC++)
    pub max_iterations: usize,
    /// MetaSAC confidence threshold (MSAC + MAGSAC++)
    pub confidence: f64,
    // source
    pub point_source: PointSource,
    pub file_path: String,
    pub loaded_pts: Vec<[f32; 3]>,
    pub loaded_colors: Vec<[u8; 3]>,  // empty = no color data in file
    pub show_rgb: bool,
    // runtime
    pub needs_reload: bool,   // show raw cloud, clear segmentation
    pub needs_run: bool,      // run segmentation on loaded_pts
    pub needs_merge: bool,    // re-run merge on existing raw_planes
    pub needs_grow: bool,     // absorb unassigned points into nearest plane
    pub needs_auto: bool,     // estimate good settings from point cloud statistics
    pub needs_recolor: bool,  // recolor leftover/background cloud without clearing segmentation
    pub leftover_entity: Option<Entity>, // the unassigned/background cloud entity
    pub leftover_pts_cache: Vec<[f32; 3]>, // unassigned points for recolor
    // Merge step parameters
    pub merge_angle_thresh: f32,  // degrees: normals closer than this → same direction
    pub merge_dist_thresh: f32,   // same plane if |d_i - d_j| < this
    pub merge_min_pts: usize,     // drop merged plane if fewer than this many inliers
    // Grow step
    pub grow_dist_thresh: f32,    // max point-to-plane dist to absorb
    pub grow_max_iters: usize,    // iterate grow+refit until convergence
    pub grow_use_normal: bool,    // filter: local normal must agree with plane normal
    pub grow_normal_angle: f32,   // degrees: max allowed angle between point normal and plane normal
    pub grow_use_curvature: bool, // filter: skip high-curvature points (edges, furniture curves)
    pub grow_max_curvature: f32,  // max local curvature to allow absorption
    pub grow_use_connectivity: bool, // filter: point must neighbor an existing inlier
    // Dollhouse
    pub dollhouse_mode: bool,     // per-frame: hide camera-facing planes
    pub dollhouse_angle: f32,     // degrees: cone half-angle for "facing"
    pub dollhouse_exterior_only: bool,   // only hide planes classified as exterior surface
    pub dollhouse_exterior_thresh: f32,  // fraction of cloud on outward side; below = exterior
    pub status: String,
    /// Raw segmentation output (before merge); inlier indices into all_pts.
    pub raw_planes: Vec<([f32; 3], f32, Vec<usize>)>,
    /// Planes after the most recent merge/grow (source for next grow step).
    pub merged_planes: Vec<([f32; 3], f32, Vec<usize>)>,
    /// (normal, d, count, centroid, is_exterior) — centroid+exterior flag for dollhouse.
    pub detected: Vec<([f32; 3], f32, usize, [f32; 3], bool)>,
    pub show_plane_list: bool,
    /// Entity handles (pts cloud + alpha mesh) for each detected plane.
    pub plane_entities: Vec<[Option<Entity>; 2]>,
    /// Per-plane visibility toggle (index matches detected).
    pub plane_visible: Vec<bool>,
    pub visibility_dirty: bool,
    pub color_planes: bool,  // false = render all plane pts in neutral gray
    /// Points used by all detected planes (for unassigned leftover rendering after merge).
    pub all_pts_cache: Vec<[f32; 3]>,
}

impl Default for VoxelPlaneState {
    fn default() -> Self {
        Self {
            n_planes: 3,
            n_inliers: 600,
            noise_std: 0.03,
            n_outliers: 200,
            seed: 42,
            auto_seed: true,
            point_size: 0.04,
            k_neighbors: 20,
            angle_thresh: 10.0,
            min_cluster_size: 30,
            dist_thresh: 0.08,
            ransac_mode: RansacMode::Simple,
            sigma_factor: 1.5,
            max_iterations: 1000,
            confidence: 0.99,
            needs_merge: false,
            needs_grow: false,
            needs_auto: false,
            needs_recolor: false,
            leftover_entity: None,
            leftover_pts_cache: Vec::new(),
            merge_angle_thresh: 5.0,
            merge_dist_thresh: 0.15,
            merge_min_pts: 200,
            grow_dist_thresh: 0.20,
            grow_max_iters: 3,
            grow_use_normal: true,
            grow_normal_angle: 30.0,
            grow_use_curvature: true,
            grow_max_curvature: 0.05,
            grow_use_connectivity: true,
            dollhouse_mode: false,
            dollhouse_angle: 60.0,
            dollhouse_exterior_only: true,
            dollhouse_exterior_thresh: 0.05,
            raw_planes: vec![],
            merged_planes: vec![],
            all_pts_cache: vec![],
            point_source: PointSource::Synthetic,
            file_path: String::new(),
            loaded_pts: Vec::new(),
            loaded_colors: Vec::new(),
            show_rgb: true,
            needs_reload: false,
            needs_run: true,
            status: String::new(),
            detected: vec![],
            show_plane_list: true,
            plane_entities: vec![],
            plane_visible: vec![],
            visibility_dirty: false,
            color_planes: true,
        }
    }
}

#[derive(Component)]
struct VpEntity;

fn on_enter(mut s: ResMut<VoxelPlaneState>) {
    if s.point_source == PointSource::Synthetic { s.needs_run = true; }
}

fn on_exit(mut commands: Commands, q: Query<Entity, With<VpEntity>>) {
    for e in &q { commands.entity(e).despawn(); }
}

fn voxel_plane_ui(mut contexts: EguiContexts, mut state: ResMut<VoxelPlaneState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("vp_panel").default_size(260.0).show(ctx, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Region-Growing + RANSAC");
            ui.small("Ling et al. ISPRS 2024");
            ui.separator();

            // Source selector
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.point_source, PointSource::Synthetic, "Synthetic");
                ui.selectable_value(&mut state.point_source, PointSource::File, "File");
            });

            if state.point_source == PointSource::File {
                ui.horizontal(|ui| {
                    ui.label("Path:");
                    ui.text_edit_singleline(&mut state.file_path);
                });
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        match read_point_cloud_file(&state.file_path) {
                            Ok(cloud) => {
                                state.loaded_colors = cloud_rgb(&cloud);
                                state.loaded_pts = cloud_xyz(&cloud);
                                let color_info = if state.loaded_colors.is_empty() { "" } else { " (RGB)" };
                                state.status = format!("Loaded {} pts{color_info} — click Segment to run", state.loaded_pts.len());
                                state.detected.clear();
                                state.needs_reload = true;
                            }
                            Err(e) => {
                                state.status = format!("Load error: {e}");
                            }
                        }
                    }
                    let seg_enabled = !state.loaded_pts.is_empty();
                    if ui.add_enabled(seg_enabled, egui::Button::new("Segment")).clicked() {
                        state.needs_run = true;
                    }
                });
                if !state.loaded_pts.is_empty() {
                    ui.horizontal(|ui| {
                        ui.small(format!("{} pts loaded", state.loaded_pts.len()));
                        if !state.loaded_colors.is_empty() {
                            if ui.checkbox(&mut state.show_rgb, "RGB").changed() {
                                if state.detected.is_empty() {
                                    state.needs_reload = true; // pre-segmentation: full reload fine
                                } else {
                                    state.needs_recolor = true; // post-segmentation: only recolor leftover
                                }
                            }
                        }
                    });
                }
                ui.separator();
            }

            egui::CollapsingHeader::new("Generation")
                .default_open(state.point_source == PointSource::Synthetic)
                .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut state.n_planes, 1..=MAX_PLANES).text("Planes"));
                ui.add(egui::Slider::new(&mut state.n_inliers, 50..=2000).text("Inliers / plane"));
                ui.add(egui::Slider::new(&mut state.noise_std, 0.001..=0.3_f32).text("Noise σ"));
                ui.add(egui::Slider::new(&mut state.n_outliers, 0..=1000).text("Outliers"));
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut state.seed, 1..=9999_u32).text("Seed"));
                    ui.checkbox(&mut state.auto_seed, "Auto");
                });
            });

            ui.separator();
            ui.horizontal(|ui| {
                let auto_enabled = !state.loaded_pts.is_empty();
                if ui.add_enabled(auto_enabled, egui::Button::new("⚙ Auto-tune"))
                    .on_hover_text("Estimate dist threshold, normal angle, and merge/grow settings from point cloud noise and density")
                    .clicked()
                {
                    state.needs_auto = true;
                }
                if !auto_enabled {
                    ui.small("(load a file first)");
                }
            });

            egui::CollapsingHeader::new("Algorithm").default_open(true).show(ui, |ui| {
                ui.add(egui::Slider::new(&mut state.k_neighbors, 5..=50_usize).text("k neighbours"));
                ui.add(egui::Slider::new(&mut state.angle_thresh, 1.0..=45.0_f32).text("Normal angle (°)"));
                ui.add(egui::Slider::new(&mut state.min_cluster_size, 5..=200_usize).text("Min cluster pts"));
                ui.add(egui::Slider::new(&mut state.dist_thresh, 0.01..=0.5_f32).text("Dist threshold"));
                ui.separator();
                ui.label("RANSAC scorer:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.ransac_mode, RansacMode::Simple, "Simple");
                    ui.selectable_value(&mut state.ransac_mode, RansacMode::Msac,   "MSAC");
                    ui.selectable_value(&mut state.ransac_mode, RansacMode::Magsac, "MAGSAC++");
                });
                if state.ransac_mode != RansacMode::Simple {
                    ui.add(
                        egui::Slider::new(&mut state.max_iterations, 100..=10000_usize)
                            .text("Max iterations"),
                    );
                    ui.add(
                        egui::Slider::new(&mut state.confidence, 0.90..=0.9999_f64)
                            .text("Confidence")
                            .custom_formatter(|v, _| format!("{:.4}", v)),
                    );
                }
                if state.ransac_mode == RansacMode::Magsac {
                    ui.add(
                        egui::Slider::new(&mut state.sigma_factor, 1.0..=10.0_f32)
                            .text("σ_max / thresh"),
                    );
                }
            });

            ui.separator();
            ui.add(egui::Slider::new(&mut state.point_size, 0.01..=0.15_f32).text("Point size"));

            ui.separator();
            if state.point_source == PointSource::Synthetic {
                if ui.button("Generate & Segment").clicked() {
                    if state.auto_seed { state.seed = lcg_next(state.seed); }
                    state.needs_run = true;
                }
            }

            // ── Merge step ──────────────────────────────────────────────────
            if !state.raw_planes.is_empty() {
                ui.separator();
                egui::CollapsingHeader::new("Merge Planes").default_open(true).show(ui, |ui| {
                    ui.add(egui::Slider::new(&mut state.merge_angle_thresh, 1.0..=30.0_f32)
                        .text("Normal angle (°)"));
                    ui.add(egui::Slider::new(&mut state.merge_dist_thresh, 0.01..=1.0_f32)
                        .text("Offset dist"));
                    ui.add(egui::Slider::new(&mut state.merge_min_pts, 10..=5000_usize)
                        .text("Min pts"));
                    if ui.button("Merge").clicked() {
                        state.needs_merge = true;
                    }
                });
            }

            // ── Grow step ────────────────────────────────────────────────────
            if !state.merged_planes.is_empty() || !state.raw_planes.is_empty() {
                ui.separator();
                egui::CollapsingHeader::new("Grow Planes").default_open(false).show(ui, |ui| {
                    ui.label("Absorbs unassigned points into nearest matching plane.");
                    ui.add(egui::Slider::new(&mut state.grow_dist_thresh, 0.01..=2.0_f32)
                        .text("Max dist"));
                    ui.add(egui::Slider::new(&mut state.grow_max_iters, 1..=10_usize)
                        .text("Iterations"));
                    ui.separator();
                    ui.label("Furniture filters:");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut state.grow_use_normal, "Normal");
                        if state.grow_use_normal {
                            ui.add(egui::Slider::new(&mut state.grow_normal_angle, 5.0..=90.0_f32)
                                .text("°"));
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut state.grow_use_curvature, "Curvature");
                        if state.grow_use_curvature {
                            ui.add(egui::Slider::new(&mut state.grow_max_curvature, 0.001..=0.5_f32)
                                .text("max"));
                        }
                    });
                    ui.checkbox(&mut state.grow_use_connectivity, "Connectivity (must touch inlier)");
                    if ui.button("Grow").clicked() {
                        state.needs_grow = true;
                    }
                });
            }

            // ── Dollhouse ────────────────────────────────────────────────────
            if !state.detected.is_empty() {
                ui.separator();
                egui::CollapsingHeader::new("Dollhouse").default_open(false).show(ui, |ui| {
                    ui.label("Hide camera-facing planes (cutaway view).");
                    ui.add(egui::Slider::new(&mut state.dollhouse_angle, 10.0..=90.0_f32)
                        .text("Facing angle (°)"));
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut state.dollhouse_exterior_only, "Exterior walls only");
                        if state.dollhouse_exterior_only {
                            ui.add(egui::Slider::new(&mut state.dollhouse_exterior_thresh, 0.01..=0.30_f32)
                                .text("outward frac"));
                        }
                    });
                    let n_ext = state.detected.iter().filter(|d| d.4).count();
                    if !state.detected.is_empty() {
                        ui.small(format!("{}/{} planes classified as exterior", n_ext, state.detected.len()));
                    }
                    ui.checkbox(&mut state.dollhouse_mode, "Live (updates with camera)");
                    if !state.dollhouse_mode {
                        if ui.button("Apply now").clicked() {
                            state.visibility_dirty = true; // triggers dollhouse_system once
                        }
                    }
                });
            }

            if !state.status.is_empty() {
                ui.separator();
                ui.small(&state.status);
            }

            if !state.detected.is_empty() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::LIGHT_BLUE,
                        format!("Detected {} planes", state.detected.len()));
                    let label = if state.show_plane_list { "▲" } else { "▼" };
                    if ui.small_button(label).clicked() {
                        state.show_plane_list = !state.show_plane_list;
                    }
                    let all_on = state.plane_visible.iter().all(|&v| v);
                    if ui.small_button(if all_on { "☐ All" } else { "☑ All" }).clicked() {
                        let new_val = !all_on;
                        for v in &mut state.plane_visible { *v = new_val; }
                        state.visibility_dirty = true;
                    }
                    if ui.small_button(if state.color_planes { "Gray" } else { "Color" }).on_hover_text(
                        if state.color_planes { "Switch to gray (no plane coloring)" } else { "Switch to per-plane colors" }
                    ).clicked() {
                        state.color_planes = !state.color_planes;
                        state.needs_run = state.detected.is_empty();
                        if !state.detected.is_empty() {
                            // Re-spawn planes with new color setting without re-segmenting.
                            // Set a flag so the scene system handles it.
                            state.needs_merge = false;
                            state.needs_grow = false;
                            state.visibility_dirty = false;
                            // Trigger a recolor by re-running whichever step produced current detected.
                            // Simplest: mark needs_merge if merged_planes exist, else needs_run.
                            if !state.merged_planes.is_empty() {
                                state.needs_merge = true;
                            } else {
                                state.needs_run = true;
                            }
                        }
                    }
                });
                if state.show_plane_list {
                    let plane_info: Vec<([f32;3], f32, usize, [f32;3], bool)> = state.detected.clone();
                    for (i, (n, d, cnt, _centroid, is_ext)) in plane_info.iter().enumerate() {
                        let [r, g, b] = PT_COLORS[i % MAX_PLANES];
                        let color = egui::Color32::from_rgb((r*255.) as u8, (g*255.) as u8, (b*255.) as u8);
                        ui.horizontal(|ui| {
                            if i < state.plane_visible.len() {
                                if ui.checkbox(&mut state.plane_visible[i], "").changed() {
                                    state.visibility_dirty = true;
                                }
                            }
                            let ext_tag = if *is_ext { " [ext]" } else { "" };
                            ui.colored_label(color,
                                format!("Plane {} | n=[{:.2},{:.2},{:.2}] d={:.3} | {} pts{}",
                                    i+1, n[0], n[1], n[2], d, cnt, ext_tag),
                            );
                        });
                    }
                }
            }
        });
    });
}

fn voxel_plane_scene(
    mut commands: Commands,
    mut state: ResMut<VoxelPlaneState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<VpEntity>>,
    mut vis_query: Query<&mut Visibility>,
) {
    // Apply per-plane visibility toggles when changed in the UI.
    if state.visibility_dirty {
        state.visibility_dirty = false;
        for (i, entities) in state.plane_entities.iter().enumerate() {
            let visible = state.plane_visible.get(i).copied().unwrap_or(true);
            let v = if visible { Visibility::Inherited } else { Visibility::Hidden };
            for &opt_e in entities.iter() {
                if let Some(e) = opt_e {
                    if let Ok(mut vis) = vis_query.get_mut(e) {
                        *vis = v;
                    }
                }
            }
        }
        return;
    }

    // Recolor the leftover cloud without disturbing plane entities or segmentation.
    if state.needs_recolor {
        state.needs_recolor = false;
        if let Some(e) = state.leftover_entity {
            let pts = state.leftover_pts_cache.clone();
            let point_size = state.point_size;
            let use_rgb = state.show_rgb && !state.loaded_colors.is_empty();
            // Find matching indices in all_pts_cache so we can look up colors.
            let new_mesh = if use_rgb && !state.loaded_colors.is_empty() {
                // Build color array for leftover pts by matching positions to loaded_pts indices.
                // Since leftover_pts_cache stores actual [f32;3] positions, we need the original
                // index to look up color. Instead, rebuild leftover indices from all_pts_cache.
                let assigned_set: std::collections::HashSet<usize> = state.plane_entities.iter()
                    .flat_map(|_| std::iter::empty::<usize>()) // placeholder — use merged/raw inliers
                    .collect();
                // Simpler: build assigned from detected planes via all_pts_cache positions.
                // Actually easiest: iterate all_pts_cache, find unassigned indices, get their colors.
                let mut inlier_flags = vec![false; state.all_pts_cache.len()];
                let source_planes = if !state.merged_planes.is_empty() {
                    &state.merged_planes
                } else {
                    &state.raw_planes
                };
                for (_, _, inliers) in source_planes {
                    for &i in inliers { if i < inlier_flags.len() { inlier_flags[i] = true; } }
                }
                let leftover_indices: Vec<usize> = (0..state.all_pts_cache.len())
                    .filter(|&i| !inlier_flags[i])
                    .collect();
                let colors: Vec<[u8;3]> = leftover_indices.iter()
                    .map(|&i| state.loaded_colors.get(i).copied().unwrap_or([180,180,180]))
                    .collect();
                make_cloud_mesh_rgb(&pts, &colors, point_size)
            } else {
                make_cloud_mesh(&pts, point_size)
            };
            commands.entity(e).insert(Mesh3d(meshes.add(new_mesh)));
            // Update material color too (gray vs white for RGB).
            let base_color = if use_rgb { Color::WHITE } else { Color::srgb(0.45, 0.45, 0.45) };
            commands.entity(e).insert(MeshMaterial3d(materials.add(StandardMaterial {
                base_color,
                unlit: true,
                ..default()
            })));
        }
        return;
    }

    // Show raw cloud after load, before segmentation.
    if state.needs_reload {
        state.needs_reload = false;
        for e in &existing { commands.entity(e).despawn(); }
        let pts = state.loaded_pts.clone();
        let use_rgb = state.show_rgb && !state.loaded_colors.is_empty();
        let mesh = if use_rgb {
            make_cloud_mesh_rgb(&pts, &state.loaded_colors, state.point_size)
        } else {
            make_cloud_mesh(&pts, state.point_size)
        };
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..default()
            })),
            Transform::default(),
            VpEntity,
        ));
        return;
    }

    // Merge step: union-find co-planar fragments and refit.
    if state.needs_merge {
        state.needs_merge = false;
        if !state.raw_planes.is_empty() {
            let merged = merge_planes(
                &state.raw_planes,
                &state.all_pts_cache,
                state.merge_angle_thresh.to_radians(),
                state.merge_dist_thresh,
                state.merge_min_pts,
            );
            for e in &existing { commands.entity(e).despawn(); }
            let all_pts_c = state.all_pts_cache.clone();
            let point_size = state.point_size;
            let n_total = all_pts_c.len();
            let leftover: Vec<[f32;3]> = {
                let mut used = vec![false; n_total];
                for (_, _, inliers) in &merged { for &i in inliers { if i < used.len() { used[i] = true; } } }
                all_pts_c.iter().enumerate().filter(|&(i,_)| !used[i]).map(|(_,&p)| p).collect()
            };
            let n_merged = merged.len();
            let exterior_thresh = state.dollhouse_exterior_thresh;
            let dist_thresh = state.dist_thresh;
            let color_planes = state.color_planes;
            let (mut det, mut ents, mut vis) = (vec![], vec![], vec![]);
            spawn_planes(&merged, &all_pts_c, &mut commands,
                &mut meshes, &mut materials, point_size, exterior_thresh, dist_thresh, color_planes,
                &mut det, &mut ents, &mut vis);
            state.detected = det; state.plane_entities = ents; state.plane_visible = vis;
            state.merged_planes = merged;
            state.leftover_pts_cache = leftover.clone();
            state.leftover_entity = if !leftover.is_empty() {
                Some(commands.spawn((
                    Mesh3d(meshes.add(make_cloud_mesh(&leftover, point_size))),
                    MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.45,0.45,0.45), unlit: true, ..default() })),
                    Transform::default(), VpEntity,
                )).id())
            } else { None };
            state.status = format!("{} pts | {} planes after merge | {} unassigned",
                n_total, n_merged, leftover.len());
        }
        return;
    }

    // Grow step: absorb unassigned leftover into nearest matching plane.
    if state.needs_grow {
        state.needs_grow = false;
        let source = if !state.merged_planes.is_empty() {
            state.merged_planes.clone()
        } else {
            state.raw_planes.iter().map(|(n,d,v)| (*n,*d,v.clone())).collect()
        };
        if !source.is_empty() {
            let max_iters = state.grow_max_iters;
            let grow_args = GrowArgs {
                dist_thresh: state.grow_dist_thresh,
                use_normal: state.grow_use_normal,
                normal_cos_thresh: state.grow_normal_angle.to_radians().cos(),
                use_curvature: state.grow_use_curvature,
                max_curvature: state.grow_max_curvature,
                use_connectivity: state.grow_use_connectivity,
            };
            let mut grown = grow_planes(&source, &state.all_pts_cache, &grow_args);
            for _ in 1..max_iters {
                let prev_count: usize = grown.iter().map(|(_,_,v)| v.len()).sum();
                grown = grow_planes(&grown, &state.all_pts_cache, &grow_args);
                let new_count: usize = grown.iter().map(|(_,_,v)| v.len()).sum();
                if new_count == prev_count { break; }
            }
            for e in &existing { commands.entity(e).despawn(); }
            let all_pts_c = state.all_pts_cache.clone();
            let point_size = state.point_size;
            let n_total = all_pts_c.len();
            let leftover: Vec<[f32;3]> = {
                let mut used = vec![false; n_total];
                for (_, _, inliers) in &grown { for &i in inliers { if i < used.len() { used[i] = true; } } }
                all_pts_c.iter().enumerate().filter(|&(i,_)| !used[i]).map(|(_,&p)| p).collect()
            };
            let n_grown = grown.len();
            let exterior_thresh = state.dollhouse_exterior_thresh;
            let dist_thresh = state.dist_thresh;
            let color_planes = state.color_planes;
            let (mut det, mut ents, mut vis) = (vec![], vec![], vec![]);
            spawn_planes(&grown, &all_pts_c, &mut commands,
                &mut meshes, &mut materials, point_size, exterior_thresh, dist_thresh, color_planes,
                &mut det, &mut ents, &mut vis);
            state.detected = det; state.plane_entities = ents; state.plane_visible = vis;
            state.merged_planes = grown;
            state.leftover_pts_cache = leftover.clone();
            state.leftover_entity = if !leftover.is_empty() {
                Some(commands.spawn((
                    Mesh3d(meshes.add(make_cloud_mesh(&leftover, point_size))),
                    MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.45,0.45,0.45), unlit: true, ..default() })),
                    Transform::default(), VpEntity,
                )).id())
            } else { None };
            state.status = format!("{} pts | {} planes after grow | {} unassigned",
                n_total, n_grown, leftover.len());
        }
        return;
    }

    // Auto-tune: estimate good algorithm settings from point cloud statistics.
    if state.needs_auto {
        state.needs_auto = false;
        if !state.loaded_pts.is_empty() {
            let tuned = auto_tune_settings(&state.loaded_pts);
            state.dist_thresh        = tuned.dist_thresh;
            state.angle_thresh       = tuned.angle_thresh;
            state.min_cluster_size   = tuned.min_cluster_size;
            state.merge_angle_thresh = tuned.merge_angle_thresh;
            state.merge_dist_thresh  = tuned.merge_dist_thresh;
            state.merge_min_pts      = tuned.merge_min_pts;
            state.grow_dist_thresh   = tuned.grow_dist_thresh;
            state.status = tuned.description;
        }
        return;
    }

    if !state.needs_run { return; }
    state.needs_run = false;

    for e in &existing { commands.entity(e).despawn(); }
    state.detected.clear();
    state.raw_planes.clear();
    state.merged_planes.clear();

    let all_pts: Vec<[f32; 3]> = match state.point_source {
        PointSource::Synthetic => synthetic_multi_plane(
            state.n_planes, state.n_inliers, state.noise_std, state.n_outliers, state.seed,
        ),
        PointSource::File => {
            if state.loaded_pts.is_empty() {
                state.status = "No file loaded.".into();
                return;
            }
            state.loaded_pts.clone()
        }
    };

    let sigma_max = (state.dist_thresh * state.sigma_factor) as f64;
    let planes = region_growing_ransac(
        &all_pts,
        state.k_neighbors,
        state.angle_thresh.to_radians(),
        state.min_cluster_size,
        state.dist_thresh,
        state.ransac_mode,
        sigma_max,
        state.max_iterations,
        state.confidence,
    );

    state.raw_planes = planes;
    state.all_pts_cache = all_pts.clone();
    let point_size = state.point_size;

    // Extract refs after storing to avoid simultaneous borrows
    let raw_planes_ref: &[([f32;3], f32, Vec<usize>)] = &state.raw_planes;
    let mut used = vec![false; all_pts.len()];
    for (_, _, inliers) in raw_planes_ref { for &i in inliers { if i < used.len() { used[i] = true; } } }
    let leftover: Vec<[f32; 3]> = all_pts.iter().enumerate()
        .filter(|&(i, _)| !used[i]).map(|(_, &p)| p).collect();

    let raw_planes_clone: Vec<([f32;3], f32, Vec<usize>)> = state.raw_planes.iter()
        .map(|(n,d,v)| (*n,*d,v.clone())).collect();
    let exterior_thresh = state.dollhouse_exterior_thresh;
    let dist_thresh = state.dist_thresh;
    let color_planes = state.color_planes;
    let (mut det, mut ents, mut vis) = (vec![], vec![], vec![]);
    spawn_planes(&raw_planes_clone, &all_pts, &mut commands,
        &mut meshes, &mut materials, point_size, exterior_thresh, dist_thresh, color_planes,
        &mut det, &mut ents, &mut vis);
    state.detected = det; state.plane_entities = ents; state.plane_visible = vis;

    state.leftover_pts_cache = leftover.clone();
    state.leftover_entity = if !leftover.is_empty() {
        Some(commands.spawn((
            Mesh3d(meshes.add(make_cloud_mesh(&leftover, point_size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.45, 0.45, 0.45),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            VpEntity,
        )).id())
    } else { None };

    state.status = format!(
        "{} pts | {} planes | {} unassigned",
        all_pts.len(), state.raw_planes.len(), leftover.len()
    );
}

#[allow(clippy::too_many_arguments)]
fn spawn_planes(
    planes: &[([f32;3], f32, Vec<usize>)],
    all_pts: &[[f32;3]],
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    point_size: f32,
    exterior_thresh: f32,
    dist_thresh: f32,   // used to exclude near-plane points from exterior count
    color_planes: bool,
    detected: &mut Vec<([f32;3], f32, usize, [f32;3], bool)>,
    plane_entities: &mut Vec<[Option<Entity>;2]>,
    plane_visible: &mut Vec<bool>,
) {
    detected.clear();
    plane_entities.clear();
    plane_visible.clear();

    for (pi, (normal, d, inliers)) in planes.iter().enumerate() {
        let ci = pi % MAX_PLANES;
        let [r, g, b] = PT_COLORS[ci];
        let [mr, mg, mb, ma] = MESH_COLORS[ci];

        let pts3d: Vec<[f32; 3]> = inliers.iter().map(|&i| all_pts[i]).collect();
        let centroid = if pts3d.is_empty() { [0.0f32;3] } else {
            let n = pts3d.len() as f32;
            [pts3d.iter().map(|p|p[0]).sum::<f32>()/n,
             pts3d.iter().map(|p|p[1]).sum::<f32>()/n,
             pts3d.iter().map(|p|p[2]).sum::<f32>()/n]
        };
        // Canonicalize normal to point toward the majority of cloud points (inward).
        // Exclude points within dist_thresh of the plane — they straddle both sides
        // due to noise and would inflate the outward count for exterior walls.
        let margin = dist_thresh * 3.0;
        let mut far_pos = 0usize;
        let mut far_neg = 0usize;
        for &p in all_pts.iter() {
            let signed = normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + d;
            if signed > margin { far_pos += 1; }
            else if signed < -margin { far_neg += 1; }
        }
        let (canonical_normal, canonical_d) = if far_pos >= far_neg {
            (*normal, *d)
        } else {
            ([-normal[0], -normal[1], -normal[2]], -*d)
        };
        // Exterior test: with canonical_normal pointing inward, the outward side is
        // canonical_normal·p + canonical_d < -margin (clearly outside, not near-plane noise).
        let total_far = (far_pos + far_neg).max(1);
        let outward_far = if far_pos >= far_neg { far_neg } else { far_pos };
        let outward_frac = outward_far as f32 / total_far as f32;
        let is_exterior = outward_frac < exterior_thresh;

        detected.push((canonical_normal, canonical_d, pts3d.len(), centroid, is_exterior));

        let pt_color = if color_planes { Color::srgb(r, g, b) } else { Color::srgb(0.65, 0.65, 0.65) };
        let pts_entity = if !pts3d.is_empty() {
            Some(commands.spawn((
                Mesh3d(meshes.add(make_cloud_mesh(&pts3d, point_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: pt_color,
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                VpEntity,
            )).id())
        } else { None };

        let alpha_pts: Vec<[f32; 3]> = if pts3d.len() > 3000 {
            let stride = pts3d.len() / 3000;
            pts3d.iter().step_by(stride).cloned().collect()
        } else {
            pts3d.clone()
        };
        let alpha_entity = make_alpha_shape_mesh(*normal, *d, &alpha_pts).map(|mesh| {
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(mr, mg, mb, ma),
                    alpha_mode: AlphaMode::Blend,
                    double_sided: true,
                    unlit: true,
                    cull_mode: None,
                    ..default()
                })),
                Transform::default(),
                VpEntity,
            )).id()
        });

        plane_entities.push([pts_entity, alpha_entity]);
        plane_visible.push(true);
        let _ = pi;
    }
}

// ─── Core pipeline ────────────────────────────────────────────────────────────

/// Run the full Ling et al. 2024 three-step pipeline on a raw point cloud.
///
/// Returns a list of `(unit_normal, d, inlier_indices)` tuples sorted by inlier
/// count descending.  The plane equation is `normal · p + d ≈ 0`.
///
/// # Steps
///
/// 1. **Normal + curvature estimation** — for each point, query 27-cell grid neighbourhood,
///    fit covariance matrix, extract smallest eigenvector (normal) and
///    `curvature = λ_min / trace`.
/// 2. **Region growing** — sort points by curvature ascending; BFS-grow from flattest
///    seeds; accept neighbour when `|n_cur · n_nb| > cos(angle_thresh)`.
/// 3. **RANSAC + aggregation** — fit a plane to each cluster via the selected RANSAC
///    mode; then sweep the entire unassigned cloud and absorb any point within
///    `dist_thresh` whose local normal also passes the angle test.
fn region_growing_ransac(
    pts: &[[f32; 3]],
    k: usize,
    angle_thresh: f32,
    min_cluster_size: usize,
    dist_thresh: f32,
    mode: RansacMode,
    sigma_max: f64,
    max_iterations: usize,
    confidence: f64,
) -> Vec<([f32; 3], f32, Vec<usize>)> {
    use std::time::Instant;
    let n = pts.len();
    if n < 3 { return vec![]; }
    eprintln!("[ling] n={n}");

    let t0 = Instant::now();
    let cell_size = estimate_cell_size(pts);
    let grid = build_grid(pts, cell_size);
    eprintln!("[ling] grid build: {:.2?}  cell_size={cell_size:.4}  cells={}", t0.elapsed(), grid.len());

    // Step 1: per-point normals and curvatures.
    let t1 = Instant::now();
    let mut normals: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    let mut curvatures: Vec<f32> = vec![f32::MAX; n];

    for i in 0..n {
        let neighbors = knn(pts, i, k, cell_size, &grid);
        if neighbors.len() < 3 { continue; }
        if let Some((nv, cv)) = pca_normal_and_curvature(pts, &neighbors) {
            normals[i] = nv;
            curvatures[i] = cv;
        }
    }
    eprintln!("[ling] step1 normals+curvature: {:.2?}", t1.elapsed());

    // Step 2: region growing sorted by curvature ascending (flattest = best seed).
    let t2 = Instant::now();
    let cos_thresh = angle_thresh.cos();
    let mut visited = vec![false; n];
    let mut sorted_idx: Vec<usize> = (0..n)
        .filter(|&i| curvatures[i] < f32::MAX)
        .collect();
    sorted_idx.sort_unstable_by(|&a, &b| curvatures[a].partial_cmp(&curvatures[b]).unwrap());

    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for &seed in &sorted_idx {
        if visited[seed] { continue; }
        visited[seed] = true;

        let mut cluster = vec![seed];
        let mut queue = VecDeque::new();
        queue.push_back(seed);

        while let Some(cur) = queue.pop_front() {
            let neighbors = knn(pts, cur, k, cell_size, &grid);
            for nb in neighbors {
                if visited[nb] { continue; }
                let nc = normals[cur];
                let nn = normals[nb];
                let dot = (nc[0]*nn[0] + nc[1]*nn[1] + nc[2]*nn[2]).abs();
                if dot < cos_thresh { continue; }
                visited[nb] = true;
                cluster.push(nb);
                queue.push_back(nb);
            }
        }

        if cluster.len() >= min_cluster_size {
            clusters.push(cluster);
        }
    }
    eprintln!("[ling] step2 region-growing: {:.2?}  clusters={}", t2.elapsed(), clusters.len());

    // Step 3: RANSAC per cluster + sweep remaining unassigned points.
    let t3 = Instant::now();
    let mut assigned = vec![false; n];
    let mut result: Vec<([f32; 3], f32, Vec<usize>)> = Vec::new();

    for cluster in clusters {
        let unassigned_count = cluster.iter().filter(|&&i| !assigned[i]).count();
        if unassigned_count < min_cluster_size { continue; }

        let unassigned: Vec<usize> = cluster.iter().cloned().filter(|&i| !assigned[i]).collect();
        let cluster_pts: Vec<[f32;3]> = unassigned.iter().map(|&i| pts[i]).collect();

        let msac_settings = || Some(MetasacSettings {
            max_iterations,
            confidence,
            ..MetasacSettings::default()
        });
        let fit = match mode {
            RansacMode::Simple => ransac_plane(&cluster_pts, dist_thresh, 200, 42)
                .map(|(n, d)| (n, d)),
            RansacMode::Msac => fit_plane_msac(&cluster_pts, dist_thresh as f64, msac_settings())
                .map(|(n, d, _)| (n, d)),
            RansacMode::Magsac => fit_plane_magsac_raw(&cluster_pts, sigma_max, msac_settings())
                .map(|(n, d, _)| (n, d)),
        };
        let (normal, d) = match fit {
            Some(nd) => nd,
            None => continue,
        };

        let mut plane_pts: Vec<usize> = Vec::new();
        for &gi in &unassigned {
            let dist = (normal[0]*pts[gi][0] + normal[1]*pts[gi][1] + normal[2]*pts[gi][2] + d).abs();
            if dist < dist_thresh {
                plane_pts.push(gi);
                assigned[gi] = true;
            }
        }

        if plane_pts.len() < min_cluster_size { continue; }

        // Aggregation: sweep ALL remaining unassigned points with distance + normal constraints.
        for i in 0..n {
            if assigned[i] { continue; }
            let dist = (normal[0]*pts[i][0] + normal[1]*pts[i][1] + normal[2]*pts[i][2] + d).abs();
            if dist >= dist_thresh { continue; }
            let dot = (normals[i][0]*normal[0] + normals[i][1]*normal[1] + normals[i][2]*normal[2]).abs();
            if dot < cos_thresh { continue; }
            plane_pts.push(i);
            assigned[i] = true;
        }

        result.push((normal, d, plane_pts));
    }
    eprintln!("[ling] step3 ransac+agg: {:.2?}  planes={}", t3.elapsed(), result.len());
    eprintln!("[ling] total: {:.2?}", t0.elapsed());

    result
}

/// 3-point RANSAC plane fitter with least-squares refinement on inliers.
///
/// Uses a 64-bit LCG (`s' = s × 6364136223846793005 + 1442695040888963407`) to
/// sample triples without external RNG dependencies.  After finding the best
/// hypothesis, refits via PCA on all inliers for sub-threshold accuracy.
/// Returns `(unit_normal, d)` such that `normal · p + d ≈ 0`, or `None`.
fn ransac_plane(
    pts: &[[f32; 3]],
    threshold: f32,
    max_iters: usize,
    seed: u64,
) -> Option<([f32; 3], f32)> {
    let n = pts.len();
    if n < 3 { return None; }

    let mut rng = seed;
    let mut lcg = move || -> usize {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };

    let mut best_count = 0usize;
    let mut best_normal = [0f32; 3];
    let mut best_d = 0f32;

    for _ in 0..max_iters {
        let i0 = lcg() % n;
        let i1 = lcg() % n;
        let i2 = lcg() % n;
        if i0 == i1 || i0 == i2 || i1 == i2 { continue; }

        let (normal, d) = match fit_plane_3pts(pts[i0], pts[i1], pts[i2]) {
            Some(nd) => nd,
            None => continue,
        };

        let count = pts.iter()
            .filter(|&&p| (normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + d).abs() < threshold)
            .count();

        if count > best_count {
            best_count = count;
            best_normal = normal;
            best_d = d;
        }
    }

    if best_count < 3 { return None; }

    // Least-squares refinement on inliers.
    let inliers: Vec<[f32;3]> = pts.iter()
        .cloned()
        .filter(|&p| (best_normal[0]*p[0] + best_normal[1]*p[1] + best_normal[2]*p[2] + best_d).abs() < threshold)
        .collect();

    fit_plane_ls(&inliers).or(Some((best_normal, best_d)))
}

fn fit_plane_3pts(a: [f32;3], b: [f32;3], c: [f32;3]) -> Option<([f32;3], f32)> {
    let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = cross3(ab, ac);
    let len = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
    if len < 1e-8 { return None; }
    let n = [n[0]/len, n[1]/len, n[2]/len];
    let d = -(n[0]*a[0] + n[1]*a[1] + n[2]*a[2]);
    Some((n, d))
}

/// Fit plane by PCA on inlier set (least-squares).
fn fit_plane_ls(pts: &[[f32;3]]) -> Option<([f32;3], f32)> {
    if pts.len() < 3 { return None; }
    let idxs: Vec<usize> = (0..pts.len()).collect();
    let (normal, _) = pca_normal_and_curvature(pts, &idxs)?;
    let cx: f32 = pts.iter().map(|p| p[0]).sum::<f32>() / pts.len() as f32;
    let cy: f32 = pts.iter().map(|p| p[1]).sum::<f32>() / pts.len() as f32;
    let cz: f32 = pts.iter().map(|p| p[2]).sum::<f32>() / pts.len() as f32;
    let d = -(normal[0]*cx + normal[1]*cy + normal[2]*cz);
    Some((normal, d))
}

// ─── Cloud helpers ────────────────────────────────────────────────────────────

/// Extract RGB colors from a PointCloud. Returns empty vec if no color fields.
fn cloud_rgb(cloud: &spatialrust_inlier::PointCloud) -> Vec<[u8; 3]> {
    use spatialrust_inlier::{FieldSemantic, PointBuffer};
    let schema = cloud.schema();
    let r_name = schema.find_semantic(FieldSemantic::ColorR).map(|f| f.name.clone());
    let g_name = schema.find_semantic(FieldSemantic::ColorG).map(|f| f.name.clone());
    let b_name = schema.find_semantic(FieldSemantic::ColorB).map(|f| f.name.clone());
    let (Some(r_name), Some(g_name), Some(b_name)) = (r_name, g_name, b_name) else {
        return vec![];
    };
    let Ok(r_buf) = cloud.field(&r_name) else { return vec![]; };
    let Ok(g_buf) = cloud.field(&g_name) else { return vec![]; };
    let Ok(b_buf) = cloud.field(&b_name) else { return vec![]; };

    let to_u8 = |buf: &PointBuffer, i: usize| -> u8 {
        match buf {
            PointBuffer::U8(v)  => v[i],
            PointBuffer::U16(v) => (v[i] >> 8) as u8,
            PointBuffer::F32(v) => (v[i].clamp(0.0, 1.0) * 255.0) as u8,
            PointBuffer::F64(v) => (v[i].clamp(0.0, 1.0) * 255.0) as u8,
            _                   => 0,
        }
    };

    (0..cloud.len()).map(|i| [to_u8(r_buf, i), to_u8(g_buf, i), to_u8(b_buf, i)]).collect()
}

/// Build a cube-splat mesh with per-vertex colors baked in.
fn make_cloud_mesh_rgb(positions: &[[f32; 3]], colors: &[[u8; 3]], size: f32) -> bevy::prelude::Mesh {
    use bevy::render::mesh::{Indices, PrimitiveTopology};
    use bevy::asset::RenderAssetUsages;

    let h = size * 0.5;
    let cap = positions.len().min(colors.len()).min(100_000);
    let n = cap;
    let mut verts:  Vec<[f32; 3]> = Vec::with_capacity(n * 24);
    let mut norms:  Vec<[f32; 3]> = Vec::with_capacity(n * 24);
    let mut cols:   Vec<[f32; 4]> = Vec::with_capacity(n * 24);
    let mut idx:    Vec<u32>      = Vec::with_capacity(n * 36);

    for (i, (&[cx, cy, cz], &[r, g, b])) in positions.iter().zip(colors.iter()).take(cap).enumerate() {
        let cr = r as f32 / 255.0;
        let cg = g as f32 / 255.0;
        let cb = b as f32 / 255.0;
        let col = [cr, cg, cb, 1.0];
        let base = (i * 24) as u32;
        verts.extend([[cx+h,cy-h,cz-h],[cx+h,cy+h,cz-h],[cx+h,cy+h,cz+h],[cx+h,cy-h,cz+h]]);
        for _ in 0..4 { norms.push([1.,0.,0.]); cols.push(col); }
        verts.extend([[cx-h,cy-h,cz+h],[cx-h,cy+h,cz+h],[cx-h,cy+h,cz-h],[cx-h,cy-h,cz-h]]);
        for _ in 0..4 { norms.push([-1.,0.,0.]); cols.push(col); }
        verts.extend([[cx-h,cy+h,cz-h],[cx-h,cy+h,cz+h],[cx+h,cy+h,cz+h],[cx+h,cy+h,cz-h]]);
        for _ in 0..4 { norms.push([0.,1.,0.]); cols.push(col); }
        verts.extend([[cx-h,cy-h,cz+h],[cx-h,cy-h,cz-h],[cx+h,cy-h,cz-h],[cx+h,cy-h,cz+h]]);
        for _ in 0..4 { norms.push([0.,-1.,0.]); cols.push(col); }
        verts.extend([[cx-h,cy-h,cz+h],[cx+h,cy-h,cz+h],[cx+h,cy+h,cz+h],[cx-h,cy+h,cz+h]]);
        for _ in 0..4 { norms.push([0.,0.,1.]); cols.push(col); }
        verts.extend([[cx+h,cy-h,cz-h],[cx-h,cy-h,cz-h],[cx-h,cy+h,cz-h],[cx+h,cy+h,cz-h]]);
        for _ in 0..4 { norms.push([0.,0.,-1.]); cols.push(col); }
        for face in 0..6u32 {
            let b = base + face * 4;
            idx.extend([b, b+1, b+2, b+2, b+3, b]);
        }
    }

    let mut mesh = bevy::prelude::Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(bevy::prelude::Mesh::ATTRIBUTE_POSITION, verts);
    mesh.insert_attribute(bevy::prelude::Mesh::ATTRIBUTE_NORMAL, norms);
    mesh.insert_attribute(bevy::prelude::Mesh::ATTRIBUTE_COLOR, cols);
    mesh.insert_indices(Indices::U32(idx));
    mesh
}

// ─── Spatial helpers ──────────────────────────────────────────────────────────

/// Choose a uniform grid cell size that targets ~8 points per cell.
///
/// With ~8 pts/cell the 3×3×3 = 27-cell neighbourhood of any query point
/// covers ~216 candidates, comfortably exceeding k = 20 even with uneven
/// density.  The formula `(bbox_volume / n)^(1/3) × 2.0` derives cell size
/// from the point density implied by the bounding box volume.
fn estimate_cell_size(pts: &[[f32; 3]]) -> f32 {
    let n = pts.len();
    if n < 2 { return 0.1; }
    // cell_size = (bbox_volume / n)^(1/3) × 2  (×2 targets ~8 pts/cell)
    let mut xmin = f32::MAX; let mut xmax = f32::MIN;
    let mut ymin = f32::MAX; let mut ymax = f32::MIN;
    let mut zmin = f32::MAX; let mut zmax = f32::MIN;
    for &p in pts {
        if p[0] < xmin { xmin = p[0]; } if p[0] > xmax { xmax = p[0]; }
        if p[1] < ymin { ymin = p[1]; } if p[1] > ymax { ymax = p[1]; }
        if p[2] < zmin { zmin = p[2]; } if p[2] > zmax { zmax = p[2]; }
    }
    let dx = (xmax - xmin).max(1e-5);
    let dy = (ymax - ymin).max(1e-5);
    let dz = (zmax - zmin).max(1e-5);
    let volume = dx * dy * dz;
    // ×2: target ~8 pts/cell, so 27 cells always cover k=20 with room to spare
    // and the hash map stays 8× smaller than at 1 pt/cell.
    ((volume / n as f32).cbrt() * 2.0).max(1e-5)
}

fn build_grid(pts: &[[f32; 3]], cell_size: f32) -> HashMap<(i32,i32,i32), Vec<usize>> {
    let inv = 1.0 / cell_size;
    let mut grid: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();
    for (i, &p) in pts.iter().enumerate() {
        let key = (
            (p[0]*inv).floor() as i32,
            (p[1]*inv).floor() as i32,
            (p[2]*inv).floor() as i32,
        );
        grid.entry(key).or_default().push(i);
    }
    grid
}

/// Approximate k-nearest neighbours via 27-cell grid search (no distance sort).
///
/// Returns up to `k` neighbour indices by collecting from the 3×3×3 cells
/// surrounding the query point's cell, stopping as soon as `k` candidates are
/// found.  Results are **not** sorted by distance — this is intentional: PCA
/// and BFS only need membership, not ordering, and sorting 27-cell contents
/// for 817k points costs ~4× more than the unsorted approach.
fn knn(
    pts: &[[f32; 3]],
    idx: usize,
    k: usize,
    cell_size: f32,
    grid: &HashMap<(i32,i32,i32), Vec<usize>>,
) -> Vec<usize> {
    let inv = 1.0 / cell_size;
    let p = pts[idx];
    let cx = (p[0]*inv).floor() as i32;
    let cy = (p[1]*inv).floor() as i32;
    let cz = (p[2]*inv).floor() as i32;
    let mut result = Vec::with_capacity(k);
    'outer: for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                if let Some(cell) = grid.get(&(cx+dx, cy+dy, cz+dz)) {
                    for &j in cell {
                        if j != idx {
                            result.push(j);
                            if result.len() >= k { break 'outer; }
                        }
                    }
                }
            }
        }
    }
    result
}

// ─── PCA ──────────────────────────────────────────────────────────────────────

/// Extract surface normal and local curvature from a neighbourhood via PCA.
///
/// Returns `(unit_normal, curvature)` where:
/// - `unit_normal` is the smallest eigenvector of the 3×3 covariance matrix
///   (the direction of least variance = surface normal).
/// - `curvature = λ_min / (λ_min + λ_mid + λ_max) = λ_min / trace`.
///   Flat surfaces → ≈ 0; curved/noisy surfaces → > 0.
///
/// ## Implementation: shift trick + power iteration
/// Direct smallest-eigenvector computation is numerically tricky.  Instead,
/// form `B = trace·I − C`; the largest eigenvector of B equals the smallest
/// eigenvector of C.  Power iteration (20 steps, seed `[0.1, 0.5, 0.9]`)
/// converges reliably for 3×3 matrices.  Curvature is recovered via the
/// Rayleigh quotient: `λ_min = v^T C v` (v already unit length).
fn pca_normal_and_curvature(pts: &[[f32; 3]], idxs: &[usize]) -> Option<([f32; 3], f32)> {
    let n = idxs.len();
    if n < 3 { return None; }
    let nf = n as f32;
    let mut cx = 0f32; let mut cy = 0f32; let mut cz = 0f32;
    for &i in idxs { cx += pts[i][0]; cy += pts[i][1]; cz += pts[i][2]; }
    cx /= nf; cy /= nf; cz /= nf;

    let mut cov = [[0f32; 3]; 3];
    for &i in idxs {
        let dx = pts[i][0] - cx;
        let dy = pts[i][1] - cy;
        let dz = pts[i][2] - cz;
        cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz;
        cov[1][1] += dy*dy; cov[1][2] += dy*dz;
        cov[2][2] += dz*dz;
    }
    cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];

    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    if trace < 1e-12 { return None; }

    // B = trace·I − C: its largest eigenvector = C's smallest (the normal).
    let b = [
        [trace - cov[0][0], -cov[0][1],        -cov[0][2]       ],
        [-cov[1][0],         trace - cov[1][1], -cov[1][2]       ],
        [-cov[2][0],        -cov[2][1],          trace - cov[2][2]],
    ];
    let v = power_iter(b, [0.1, 0.5, 0.9]);
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len < 1e-6 { return None; }
    let normal = [v[0]/len, v[1]/len, v[2]/len];

    // Curvature = λ_min / trace = (vᵀ C v) / trace.
    let cv = matvec(cov, normal);
    let lambda_min = normal[0]*cv[0] + normal[1]*cv[1] + normal[2]*cv[2];
    let curvature = (lambda_min / trace).max(0.0);

    Some((normal, curvature))
}

fn power_iter(m: [[f32;3];3], init: [f32;3]) -> [f32;3] {
    let mut v = normalize3(init);
    for _ in 0..20 {
        let w = matvec(m, v);
        let len = (w[0]*w[0]+w[1]*w[1]+w[2]*w[2]).sqrt();
        if len < 1e-10 { break; }
        v = [w[0]/len, w[1]/len, w[2]/len];
    }
    v
}

fn matvec(m: [[f32;3];3], v: [f32;3]) -> [f32;3] {
    [
        m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
        m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
        m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2],
    ]
}

fn normalize3(v: [f32;3]) -> [f32;3] {
    let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt().max(1e-8);
    [v[0]/len, v[1]/len, v[2]/len]
}

// ─── Merge step ───────────────────────────────────────────────────────────────

/// Union-find merge of co-planar plane fragments.
///
/// Region growing + RANSAC tends to over-segment large surfaces into several
/// smaller planes (scan occlusion, density variations, doorways).  This step
/// collapses fragments that describe the same physical surface.
///
/// ## Merge criterion
/// Two planes `i` and `j` are merged when **both**:
/// - `|n_i · n_j| > cos(angle_thresh)` — normals are nearly parallel
/// - `|d_i − d_j| < dist_thresh` — same signed offset from origin
///   (d_j is sign-adjusted to match the n_i orientation before comparing)
///
/// Union-find with path compression collects transitively connected groups.
/// Each group is refit via PCA on the union of inlier points, then dropped
/// if the combined count is below `min_pts`.
fn merge_planes(
    planes: &[([f32;3], f32, Vec<usize>)],
    all_pts: &[[f32;3]],
    angle_thresh: f32,   // radians
    dist_thresh: f32,
    min_pts: usize,
) -> Vec<([f32;3], f32, Vec<usize>)> {
    let n = planes.len();
    if n == 0 { return vec![]; }

    let cos_thresh = angle_thresh.cos();

    // Union-find
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
        x
    }

    for i in 0..n {
        for j in i+1..n {
            let ni = planes[i].0;
            let nj = planes[j].0;
            let dot = (ni[0]*nj[0] + ni[1]*nj[1] + ni[2]*nj[2]).abs();
            if dot < cos_thresh { continue; }
            // Both d values should be in the same normal direction; if normals
            // were flipped, offset signs differ — normalise by taking abs diff
            // only when dot is high (normals nearly parallel, same sign).
            let di = planes[i].1;
            let dj = if planes[i].0[0]*planes[j].0[0] + planes[i].0[1]*planes[j].0[1] + planes[i].0[2]*planes[j].0[2] >= 0.0 {
                planes[j].1
            } else {
                -planes[j].1
            };
            if (di - dj).abs() > dist_thresh { continue; }
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri != rj { parent[ri] = rj; }
        }
    }

    // Collect groups
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    let mut result = Vec::new();
    for (_, members) in groups {
        // Gather all inlier indices
        let mut all_inliers: Vec<usize> = members.iter()
            .flat_map(|&pi| planes[pi].2.iter().cloned())
            .collect();
        all_inliers.sort_unstable();
        all_inliers.dedup();
        if all_inliers.len() < min_pts { continue; }

        // Refit via PCA on combined inlier points
        let pts3d: Vec<[f32;3]> = all_inliers.iter().map(|&i| all_pts[i]).collect();
        let idx: Vec<usize> = (0..pts3d.len()).collect();
        let (normal, _) = match pca_normal_and_curvature(&pts3d, &idx) {
            Some(r) => r,
            None => continue,
        };
        let cx: f32 = pts3d.iter().map(|p| p[0]).sum::<f32>() / pts3d.len() as f32;
        let cy: f32 = pts3d.iter().map(|p| p[1]).sum::<f32>() / pts3d.len() as f32;
        let cz: f32 = pts3d.iter().map(|p| p[2]).sum::<f32>() / pts3d.len() as f32;
        let d = -(normal[0]*cx + normal[1]*cy + normal[2]*cz);
        result.push((normal, d, all_inliers));
    }

    // Sort largest first
    result.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
    result
}

// ─── Grow step ────────────────────────────────────────────────────────────────

struct GrowArgs {
    dist_thresh:       f32,
    use_normal:        bool,
    normal_cos_thresh: f32,  // cos(grow_normal_angle)
    use_curvature:     bool,
    max_curvature:     f32,
    use_connectivity:  bool,
}

/// Absorb unassigned leftover points into their nearest geometrically-matching plane.
///
/// After merge, many wall/floor/ceiling fragments remain unassigned because
/// region growing was seeded from flat regions and stopped at noise boundaries.
/// Grow sweeps these leftovers and assigns each to the nearest plane within
/// `dist_thresh`, then refits each plane via PCA.  The process repeats for
/// up to `grow_max_iters` passes (stopping early when the inlier count
/// stabilises), which lets fragments reachable only via an intermediate
/// neighbour get absorbed in later passes.
///
/// ## Furniture filters (Rabbani et al. 2006 / PCL region growing)
///
/// A plain distance-only absorb will pull in furniture close to walls.
/// Three independently toggleable filters guard against this:
///
/// - **Normal filter**: computes the local PCA normal of the candidate point
///   and rejects it if `|pt_normal · plane_normal| < cos(normal_angle)`.
///   Effective for furniture oriented differently from the target plane
///   (table top near a vertical wall), but not for flush furniture.
///
/// - **Curvature filter**: rejects points whose local curvature exceeds
///   `max_curvature`.  Furniture edges and rounded surfaces have high
///   curvature; flat wall/floor regions do not.  Can over-reject wall corners.
///
/// - **Connectivity filter**: rejects the point if none of its 27-cell grid
///   neighbours is an existing inlier of the target plane.  Prevents floating
///   furniture from bridging across a scan gap to a wall.  The inlier sets are
///   built **before** the grow pass so a furniture chain cannot relay across
///   the cloud point-by-point during a single iteration.
///   **Limitation**: furniture physically touching a wall passes this filter.
fn grow_planes(
    planes: &[([f32;3], f32, Vec<usize>)],
    all_pts: &[[f32;3]],
    args: &GrowArgs,
) -> Vec<([f32;3], f32, Vec<usize>)> {
    let n_pts = all_pts.len();
    let mut assigned = vec![false; n_pts];
    let mut extended: Vec<Vec<usize>> = planes.iter().map(|(_, _, v)| {
        for &i in v { if i < n_pts { assigned[i] = true; } }
        v.clone()
    }).collect();

    // Precompute per-point normals + curvatures once (needed by normal/curvature filters).
    let need_normals = args.use_normal || args.use_curvature;
    let (pt_normals, pt_curvatures, grid, cell_size) = if need_normals {
        let cs = estimate_cell_size(all_pts);
        let g = build_grid(all_pts, cs);
        let mut normals = vec![[0f32; 3]; n_pts];
        let mut curvatures = vec![f32::MAX; n_pts];
        for i in 0..n_pts {
            if assigned[i] { continue; } // only need unassigned
            let neighbors = knn(all_pts, i, 20, cs, &g);
            if let Some((nv, cv)) = pca_normal_and_curvature(all_pts, &neighbors) {
                normals[i] = nv;
                curvatures[i] = cv;
            }
        }
        (normals, curvatures, g, cs)
    } else {
        // Build grid anyway if connectivity filter is on.
        let cs = estimate_cell_size(all_pts);
        let g = if args.use_connectivity { build_grid(all_pts, cs) } else { HashMap::new() };
        (vec![[0f32;3]; n_pts], vec![f32::MAX; n_pts], g, cs)
    };

    // Per-plane inlier sets for fast connectivity lookup (built from *initial* inliers only).
    let inlier_sets: Vec<std::collections::HashSet<usize>> = if args.use_connectivity {
        planes.iter().map(|(_, _, v)| v.iter().cloned().collect()).collect()
    } else {
        vec![]
    };

    let inv = 1.0 / cell_size;
    for (idx, &pt) in all_pts.iter().enumerate() {
        if assigned[idx] { continue; }

        // Curvature gate.
        if args.use_curvature && pt_curvatures[idx] > args.max_curvature { continue; }

        let mut best_dist = args.dist_thresh;
        let mut best_plane: Option<usize> = None;

        for (pi, (pn, d, _)) in planes.iter().enumerate() {
            // Distance gate.
            let dist = (pn[0]*pt[0] + pn[1]*pt[1] + pn[2]*pt[2] + d).abs();
            if dist >= best_dist { continue; }

            // Normal gate.
            if args.use_normal {
                let dot = (pt_normals[idx][0]*pn[0]
                         + pt_normals[idx][1]*pn[1]
                         + pt_normals[idx][2]*pn[2]).abs();
                if dot < args.normal_cos_thresh { continue; }
            }

            // Connectivity gate: at least one 27-cell neighbor must already be in this plane.
            if args.use_connectivity {
                let cx = (pt[0]*inv).floor() as i32;
                let cy = (pt[1]*inv).floor() as i32;
                let cz = (pt[2]*inv).floor() as i32;
                let mut connected = false;
                'conn: for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dz in -1i32..=1 {
                            if let Some(cell) = grid.get(&(cx+dx, cy+dy, cz+dz)) {
                                for &nb in cell {
                                    if inlier_sets[pi].contains(&nb) {
                                        connected = true;
                                        break 'conn;
                                    }
                                }
                            }
                        }
                    }
                }
                if !connected { continue; }
            }

            best_dist = dist;
            best_plane = Some(pi);
        }

        if let Some(pi) = best_plane {
            extended[pi].push(idx);
        }
    }

    // Refit each plane via PCA on expanded inlier set.
    let mut result = Vec::with_capacity(planes.len());
    for (pi, _) in planes.iter().enumerate() {
        let inliers = std::mem::take(&mut extended[pi]);
        if inliers.len() < 3 { continue; }
        let pts3d: Vec<[f32;3]> = inliers.iter().map(|&i| all_pts[i]).collect();
        let idxs: Vec<usize> = (0..pts3d.len()).collect();
        let (normal, _) = match pca_normal_and_curvature(&pts3d, &idxs) {
            Some(r) => r,
            None => continue,
        };
        let cx = pts3d.iter().map(|p|p[0]).sum::<f32>() / pts3d.len() as f32;
        let cy = pts3d.iter().map(|p|p[1]).sum::<f32>() / pts3d.len() as f32;
        let cz = pts3d.iter().map(|p|p[2]).sum::<f32>() / pts3d.len() as f32;
        let d = -(normal[0]*cx + normal[1]*cy + normal[2]*cz);
        result.push((normal, d, inliers));
    }

    result.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
    result
}

// ─── Auto-tune ────────────────────────────────────────────────────────────────

struct TunedSettings {
    dist_thresh:        f32,
    angle_thresh:       f32,  // degrees
    min_cluster_size:   usize,
    merge_angle_thresh: f32,  // degrees
    merge_dist_thresh:  f32,
    merge_min_pts:      usize,
    grow_dist_thresh:   f32,
    description:        String,
}

/// Estimate good algorithm parameters from point cloud statistics.
///
/// Strides the input to ~2000 sample points, builds a small spatial grid,
/// then computes 20-NN for each sample to extract:
///
/// - **`noise_sigma`** = median of per-point plane residuals.
///   Each residual is `|n · p + d|` where `(n, d)` is the PCA plane fitted
///   to the point's 20 neighbours.  The median is robust to outliers and
///   approximates the sensor measurement noise floor.
///
/// - **`point_spacing`** = median nearest-neighbour distance, a proxy for
///   surface point density.
///
/// Derived thresholds:
/// ```text
/// dist_thresh        = noise_sigma × 4       → RANSAC inlier band (4σ)
/// angle_thresh       = 15° + σ × 200°        → looser for noisy sensors
/// min_cluster_size   = 0.5 m² × density      → ~half a square metre
/// merge_angle_thresh = angle_thresh × 0.4    → joins same-wall fragments
/// merge_dist_thresh  = dist_thresh × 3       → accounts for wall offsets
/// grow_dist_thresh   = dist_thresh × 2       → mops up near-plane leftovers
/// ```
fn auto_tune_settings(pts: &[[f32; 3]]) -> TunedSettings {
    let n = pts.len();
    // Stride to at most 2000 sample points for speed.
    let stride = (n / 2000).max(1);
    let samples: Vec<[f32; 3]> = pts.iter().step_by(stride).cloned().collect();
    let ns = samples.len();

    let cell_size = estimate_cell_size(&samples);
    let grid = build_grid(&samples, cell_size);

    let k = 20usize;
    let mut residuals: Vec<f32> = Vec::with_capacity(ns);
    let mut nn_dists:  Vec<f32> = Vec::with_capacity(ns);

    for i in 0..ns {
        let neighbors = knn(&samples, i, k, cell_size, &grid);
        if neighbors.len() < 6 { continue; }

        // Nearest-neighbor distance (proxy for point spacing).
        let p = samples[i];
        let mut min_d = f32::MAX;
        for &j in &neighbors {
            let q = samples[j];
            let d = ((p[0]-q[0]).powi(2) + (p[1]-q[1]).powi(2) + (p[2]-q[2]).powi(2)).sqrt();
            if d < min_d { min_d = d; }
        }
        if min_d < f32::MAX { nn_dists.push(min_d); }

        // Local plane residual.
        if let Some((normal, _)) = pca_normal_and_curvature(&samples, &neighbors) {
            let cx = neighbors.iter().map(|&j| samples[j][0]).sum::<f32>() / neighbors.len() as f32;
            let cy = neighbors.iter().map(|&j| samples[j][1]).sum::<f32>() / neighbors.len() as f32;
            let cz = neighbors.iter().map(|&j| samples[j][2]).sum::<f32>() / neighbors.len() as f32;
            let d = -(normal[0]*cx + normal[1]*cy + normal[2]*cz);
            let res = (normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + d).abs();
            residuals.push(res);
        }
    }

    // Median helpers.
    let median = |v: &mut Vec<f32>| -> f32 {
        if v.is_empty() { return 0.01; }
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    let noise_sigma = median(&mut residuals).max(1e-4);
    let point_spacing = median(&mut nn_dists).max(1e-4);

    // dist_thresh: 3–4× noise so clean inliers pass, outliers don't.
    let dist_thresh = (noise_sigma * 4.0).clamp(0.01, 0.5);

    // normal_angle: loose enough for noisy surfaces (~2–3× noise scaled to angle).
    // Empirically: steeper noise → looser angle. Use a fixed 15° base + noise scaling.
    let angle_thresh = (15.0_f32 + noise_sigma * 200.0).clamp(5.0, 45.0);

    // min_cluster_size: ~0.5m² surface at the estimated point density.
    // density = 1 / point_spacing² pts/m²; 0.5m² → 0.5 / spacing²
    let density = 1.0 / (point_spacing * point_spacing);
    let min_cluster_size = ((density * 0.5) as usize).clamp(20, 2000);

    // merge thresholds: looser than segmentation to join fragments of same wall.
    let merge_angle_thresh = (angle_thresh * 0.4).clamp(2.0, 10.0);
    let merge_dist_thresh  = (dist_thresh * 3.0).clamp(0.05, 1.0);
    let merge_min_pts      = (min_cluster_size / 4).clamp(50, 1000);

    // grow threshold: slightly looser than dist_thresh to mop up near-plane points.
    let grow_dist_thresh = (dist_thresh * 2.0).clamp(0.02, 1.0);

    TunedSettings {
        dist_thresh,
        angle_thresh,
        min_cluster_size,
        merge_angle_thresh,
        merge_dist_thresh,
        merge_min_pts,
        grow_dist_thresh,
        description: format!(
            "Auto-tune: noise_σ={:.4}m  spacing={:.4}m → dist={:.3}  angle={:.1}°  min_pts={}  grow_dist={:.3}",
            noise_sigma, point_spacing, dist_thresh, angle_thresh, min_cluster_size, grow_dist_thresh
        ),
    }
}

// ─── Dollhouse system ─────────────────────────────────────────────────────────

fn dollhouse_system(
    state: Res<VoxelPlaneState>,
    cam_query: Query<&Transform, With<OrbitCamera>>,
    mut vis_query: Query<&mut Visibility>,
) {
    if !state.dollhouse_mode { return; }
    let Ok(cam_tf) = cam_query.single() else { return };
    let cam_pos = cam_tf.translation;
    let cos_thresh = state.dollhouse_angle.to_radians().cos();

    for (i, &(normal, _, _, centroid, is_exterior)) in state.detected.iter().enumerate() {
        // Exterior-only mode: skip interior planes entirely (leave them always visible).
        if state.dollhouse_exterior_only && !is_exterior { continue; }

        let to_cam = bevy::math::Vec3::new(
            cam_pos.x - centroid[0],
            cam_pos.y - centroid[1],
            cam_pos.z - centroid[2],
        );
        let len = to_cam.length();
        if len < 1e-6 { continue; }
        let to_cam_n = to_cam / len;
        // `normal` is canonicalized inward (toward cloud center).
        // If the inward normal points AWAY from camera (dot < 0), camera is
        // on the outside of that face — that's the face we want to hide.
        let n = bevy::math::Vec3::from(normal);
        let dot = n.dot(to_cam_n);
        // Hide when camera is on the exterior side and roughly facing it
        let facing = dot < -cos_thresh;
        // Override visibility only when dollhouse says to hide — don't fight the
        // per-plane checkbox (which sets visibility_dirty and writes plane_visible).
        let v = if facing { Visibility::Hidden } else { Visibility::Inherited };
        if let Some(entities) = state.plane_entities.get(i) {
            for &opt_e in entities.iter() {
                if let Some(e) = opt_e {
                    if let Ok(mut vis) = vis_query.get_mut(e) {
                        *vis = v;
                    }
                }
            }
        }
    }
}

// ─── Synthetic scene ──────────────────────────────────────────────────────────

fn synthetic_multi_plane(
    n_planes: usize,
    n_inliers: usize,
    noise_std: f32,
    n_outliers: usize,
    seed: u32,
) -> Vec<[f32; 3]> {
    let mut s = seed as u64;
    let mut rng = move || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let mut out = Vec::new();
    for pi in 0..n_planes.min(MAX_PLANES) {
        let normal = loop {
            let (x, y, z) = (rng(), rng(), rng());
            let len = (x*x+y*y+z*z).sqrt();
            if len > 0.01 && len <= 1.0 { break normalize3([x, y, z]); }
        };
        let offset = pi as f32 * 2.5;
        let p0 = [-offset*normal[0], -offset*normal[1], -offset*normal[2]];
        let up = if normal[2].abs() < 0.9 { [0f32,0.,1.] } else { [1.,0.,0.] };
        let t1 = normalize3(cross3(normal, up));
        let t2 = cross3(normal, t1);
        for _ in 0..n_inliers {
            let u = rng() * 3.0;
            let v = rng() * 3.0;
            let noise = rng() * noise_std;
            out.push([
                p0[0] + u*t1[0] + v*t2[0] + noise*normal[0],
                p0[1] + u*t1[1] + v*t2[1] + noise*normal[1],
                p0[2] + u*t1[2] + v*t2[2] + noise*normal[2],
            ]);
        }
    }
    for _ in 0..n_outliers {
        out.push([rng()*4., rng()*4., rng()*4.]);
    }
    out
}

fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn lcg_next(seed: u32) -> u32 {
    let s = (seed as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((s >> 33) as u32 % 9998) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_3_planes() {
        let pts = synthetic_multi_plane(3, 600, 0.03, 200, 42);
        println!("Total points: {}", pts.len());

        let planes = region_growing_ransac(
            &pts,
            20,
            10f32.to_radians(),
            30,
            0.08,
            RansacMode::Simple,
            0.0,
            1000,
            0.99,
        );

        println!("Detected {} planes:", planes.len());
        for (i, (n, d, inliers)) in planes.iter().enumerate() {
            println!("  Plane {}: n=[{:.3},{:.3},{:.3}] d={:.3}  pts={}", i+1, n[0], n[1], n[2], d, inliers.len());
        }

        assert_eq!(planes.len(), 3, "should find 3 planes, found {}", planes.len());
        for (i, (_, _, inliers)) in planes.iter().enumerate() {
            assert!(inliers.len() >= 400, "plane {} has only {} pts (want >=400)", i+1, inliers.len());
        }
    }

    #[test]
    fn smoke_3_planes_msac() {
        let pts = synthetic_multi_plane(3, 600, 0.03, 200, 42);
        let planes = region_growing_ransac(&pts, 20, 10f32.to_radians(), 30, 0.08, RansacMode::Msac, 0.0, 1000, 0.99);
        println!("MSAC: {} planes", planes.len());
        for (i, (n, d, inliers)) in planes.iter().enumerate() {
            println!("  Plane {}: n=[{:.3},{:.3},{:.3}] d={:.3}  pts={}", i+1, n[0], n[1], n[2], d, inliers.len());
        }
        assert_eq!(planes.len(), 3, "MSAC should find 3 planes, found {}", planes.len());
        for (i, (_, _, inliers)) in planes.iter().enumerate() {
            assert!(inliers.len() >= 400, "MSAC plane {} has only {} pts", i+1, inliers.len());
        }
    }

    #[test]
    fn smoke_3_planes_magsac() {
        let pts = synthetic_multi_plane(3, 600, 0.03, 200, 42);
        let sigma_max = (0.08f32 * 1.5) as f64;
        let planes = region_growing_ransac(&pts, 20, 10f32.to_radians(), 30, 0.08, RansacMode::Magsac, sigma_max, 1000, 0.99);
        println!("MAGSAC++: {} planes", planes.len());
        for (i, (n, d, inliers)) in planes.iter().enumerate() {
            println!("  Plane {}: n=[{:.3},{:.3},{:.3}] d={:.3}  pts={}", i+1, n[0], n[1], n[2], d, inliers.len());
        }
        assert_eq!(planes.len(), 3, "MAGSAC++ should find 3 planes, found {}", planes.len());
        for (i, (_, _, inliers)) in planes.iter().enumerate() {
            assert!(inliers.len() >= 400, "MAGSAC++ plane {} has only {} pts", i+1, inliers.len());
        }
    }
}
