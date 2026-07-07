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

use crate::AppDemo;
use crate::plane_demo::{make_cloud_mesh, make_alpha_shape_mesh, cloud_xyz};
use crate::OrbitCamera;
use spatialrust_inlier::{
    fit_plane_msac, fit_plane_magsac_raw, MetasacSettings,
    io::read_point_cloud_file,
    spatial_grid::{estimate_cell_size, build_grid, knn},
    normals::{pca_normal_and_curvature, normalize3, cross3, fit_plane_3pts, fit_plane_ls},
    auto_tune::{auto_tune_settings, TunedSettings},
    region_growing::{region_growing_ransac, RansacMode},
    plane_estimation::{GlobalPlanePeeling, ManhattanPlanes, PlaneEstimator, RegionGrowing},
    plane_ops::{merge_planes, grow_planes, GrowArgs},
    dollhouse::classify_plane,
    building::{reconstruct_building, BuildingParams},
};

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

/// A plane result: (unit normal, plane offset d, inlier indices into all_pts).
type PlaneVec = Vec<([f32; 3], f32, Vec<usize>)>;

/// Delivered by the background segmentation once it finishes.
struct SegOut {
    planes: PlaneVec,
    /// The point set the plane indices refer to (== loaded_pts for estimator
    /// methods; the downsampled cloud for the Building method).
    all_pts: Vec<[f32; 3]>,
    /// `Some` only for the Building method: (per-plane exterior flags from the
    /// footprint classifier, estimated up). Carrying these lets the dollhouse use
    /// the footprint exterior + the true up instead of `classify_plane`'s
    /// side-test + `estimate_up` (which would return a horizontal vector, since
    /// building planes are walls only).
    building: Option<(Vec<bool>, [f32; 3])>,
}

/// An in-flight background segmentation. The heavy compute runs off the render
/// thread so the UI keeps painting and can show live progress.
struct SegJob {
    /// (fraction in [0,1], phase label), updated by the compute thread.
    progress: Arc<Mutex<(f32, String)>>,
    /// Delivers the result once the compute finishes. Wrapped in a Mutex so
    /// `SegJob` is `Sync` (a bevy Resource requirement); `mpsc::Receiver` is
    /// `Send` but not `Sync`.
    rx: Mutex<Receiver<SegOut>>,
}

/// Run `f` off the render thread when the platform supports it, so segmentation
/// doesn't block rendering. Native uses an OS thread; the wasm "threads" build
/// uses the rayon worker pool (std::thread can't spawn Web Workers). Plain wasm
/// has no threads, so it runs inline (the UI freezes for the compute's duration).
#[cfg(not(target_arch = "wasm32"))]
fn spawn_compute<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f);
}
#[cfg(all(target_arch = "wasm32", feature = "threads"))]
fn spawn_compute<F: FnOnce() + Send + 'static>(f: F) {
    rayon::spawn(f);
}
#[cfg(all(target_arch = "wasm32", not(feature = "threads")))]
fn spawn_compute<F: FnOnce() + Send + 'static>(f: F) {
    f();
}

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

/// Which plane-estimation method to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneMethod {
    /// Region-growing + RANSAC (fragments on holey/variable-density clouds).
    RegionGrowing,
    /// Global dominant-plane peeling (high coverage; can slice through).
    Peeling,
    /// Manhattan-frame extraction — planes in all orientations. Best for
    /// buildings; feeds the dollhouse clean, correctly-oriented walls.
    Manhattan,
    /// Full building pipeline: align → split storeys → per-floor walls →
    /// footprint-based exterior classification. Downsamples internally and
    /// returns walls with exterior labels the dollhouse uses directly.
    Building,
}

impl PlaneMethod {
    fn label(self) -> &'static str {
        match self {
            PlaneMethod::RegionGrowing => "Region-grow",
            PlaneMethod::Peeling => "Peeling",
            PlaneMethod::Manhattan => "Manhattan",
            PlaneMethod::Building => "Building",
        }
    }
}

/// How to color the points belonging to detected planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneColorMode {
    /// Per-plane palette color (default).
    Palette,
    /// Each point keeps its original point-cloud RGB (file sources only).
    Original,
    /// All plane points neutral gray.
    Gray,
}

impl PlaneColorMode {
    /// Cycle to the next mode. `has_rgb` gates the `Original` mode so it is
    /// skipped for synthetic/RGB-less clouds where it would silently fall back.
    fn next(self, has_rgb: bool) -> Self {
        match self {
            PlaneColorMode::Palette => if has_rgb { PlaneColorMode::Original } else { PlaneColorMode::Gray },
            PlaneColorMode::Original => PlaneColorMode::Gray,
            PlaneColorMode::Gray => PlaneColorMode::Palette,
        }
    }

    /// Short label for the cycle button.
    fn label(self) -> &'static str {
        match self {
            PlaneColorMode::Palette => "Palette",
            PlaneColorMode::Original => "RGB",
            PlaneColorMode::Gray => "Gray",
        }
    }
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

/// Result of the browser "Load" async fetch. Native builds read the file
/// synchronously via `read_point_cloud_file` and never use this.
#[cfg(target_arch = "wasm32")]
#[derive(Default)]
enum Fetch {
    #[default]
    Idle,
    Pending,
    Done(Vec<u8>),
    Failed(String),
}

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
    pub plane_method: PlaneMethod,
    /// Estimated up direction (dominant plane normal); used to tell walls from
    /// floors so the dollhouse only hides walls.
    pub up_dir: [f32; 3],
    /// sigma_max for MAGSAC++ (multiples of dist_thresh; UI shows the multiplier)
    pub sigma_factor: f32,
    /// MetaSAC max iterations (MSAC + MAGSAC++)
    pub max_iterations: usize,
    /// MetaSAC confidence threshold (MSAC + MAGSAC++)
    pub confidence: f64,
    // source
    pub point_source: PointSource,
    pub file_path: String,
    /// Shared slot for the browser async fetch of `file_path` (wasm only).
    #[cfg(target_arch = "wasm32")]
    pub fetch: std::sync::Arc<std::sync::Mutex<Fetch>>,
    pub loaded_pts: Vec<[f32; 3]>,
    pub loaded_colors: Vec<[u8; 3]>,  // empty = no color data in file
    pub show_rgb: bool,
    // runtime
    /// In-flight background segmentation (None when idle). Not in Default.
    pub seg_job: Option<SegJob>,
    pub seg_active: bool,     // a segmentation is running (drives the progress bar)
    pub seg_progress: f32,    // last reported fraction in [0,1]
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
    /// Per-`raw_planes` exterior override from the Building method's footprint
    /// classifier. INVARIANT: these labels belong to the CURRENT `detected`;
    /// empty means "use classify_plane's side-test". Set only in the building
    /// Segment path; cleared in every non-building regen (Segment/Merge/Grow).
    pub building_exterior: Vec<bool>,
    pub show_plane_list: bool,
    /// Entity handles (pts cloud + alpha mesh) for each detected plane.
    pub plane_entities: Vec<[Option<Entity>; 2]>,
    /// Per-plane visibility toggle (index matches detected).
    pub plane_visible: Vec<bool>,
    pub visibility_dirty: bool,
    pub plane_color_mode: PlaneColorMode,  // how plane inlier points are colored
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
            plane_method: PlaneMethod::Manhattan,
            up_dir: [0.0, 1.0, 0.0],
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
            file_path: "SALON.ply".into(),
            #[cfg(target_arch = "wasm32")]
            fetch: std::sync::Arc::new(std::sync::Mutex::new(Fetch::Idle)),
            loaded_pts: Vec::new(),
            loaded_colors: Vec::new(),
            show_rgb: true,
            seg_job: None,
            seg_active: false,
            seg_progress: 0.0,
            needs_reload: false,
            needs_run: true,
            status: String::new(),
            detected: vec![],
            building_exterior: vec![],
            show_plane_list: true,
            plane_entities: vec![],
            plane_visible: vec![],
            visibility_dirty: false,
            plane_color_mode: PlaneColorMode::Palette,
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
    // Browser: pick up a completed async fetch of the point-cloud asset.
    #[cfg(target_arch = "wasm32")]
    poll_fetch(&mut state);
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

            // Plane-estimation method selector.
            ui.horizontal(|ui| {
                ui.label("Method:");
                for m in [PlaneMethod::RegionGrowing, PlaneMethod::Peeling, PlaneMethod::Manhattan, PlaneMethod::Building] {
                    ui.selectable_value(&mut state.plane_method, m, m.label());
                }
            });

            if state.point_source == PointSource::File {
                ui.horizontal(|ui| {
                    ui.label("Path:");
                    ui.text_edit_singleline(&mut state.file_path);
                });
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        #[cfg(not(target_arch = "wasm32"))]
                        match read_point_cloud_file(&state.file_path) {
                            Ok(cloud) => state.set_cloud(&cloud),
                            Err(e) => state.status = format!("Load error: {e}"),
                        }
                        // Browser: fetch the bundled asset over HTTP; poll_fetch
                        // (top of this fn) parses it once the bytes arrive.
                        #[cfg(target_arch = "wasm32")]
                        {
                            let url = state.file_path.clone();
                            start_fetch(&state.fetch, &url);
                            state.status = format!("Fetching {url}…");
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

            if state.seg_active {
                ui.separator();
                ui.add(egui::ProgressBar::new(state.seg_progress).show_percentage().animate(true));
                // Keep repainting so the bar advances while the worker runs.
                ui.ctx().request_repaint();
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
                    if ui.small_button(state.plane_color_mode.label()).on_hover_text(
                        "Cycle plane coloring: Palette → RGB (original) → Gray"
                    ).clicked() {
                        let has_rgb = !state.loaded_colors.is_empty();
                        state.plane_color_mode = state.plane_color_mode.next(has_rgb);
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
            let color_mode = state.plane_color_mode;
            let plane_colors = state.loaded_colors.clone();
            let (mut det, mut ents, mut vis) = (vec![], vec![], vec![]);
            spawn_planes(&merged, &all_pts_c, &mut commands,
                &mut meshes, &mut materials, point_size, exterior_thresh, dist_thresh, color_mode, &plane_colors,
                None, &mut det, &mut ents, &mut vis);
            state.detected = det; state.plane_entities = ents; state.plane_visible = vis;
            state.building_exterior.clear(); // merge re-runs side-test; labels no longer belong to detected
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
            let color_mode = state.plane_color_mode;
            let plane_colors = state.loaded_colors.clone();
            let (mut det, mut ents, mut vis) = (vec![], vec![], vec![]);
            spawn_planes(&grown, &all_pts_c, &mut commands,
                &mut meshes, &mut materials, point_size, exterior_thresh, dist_thresh, color_mode, &plane_colors,
                None, &mut det, &mut ents, &mut vis);
            state.detected = det; state.plane_entities = ents; state.plane_visible = vis;
            state.building_exterior.clear(); // grow re-runs side-test; labels no longer belong to detected
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

    // Poll an in-flight background segmentation started on a previous frame.
    if state.seg_job.is_some() {
        // Snapshot progress (owned) before mutating state, to end the borrow.
        let prog = state.seg_job.as_ref()
            .and_then(|j| j.progress.lock().ok().map(|p| (p.0, p.1.clone())));
        if let Some((frac, phase)) = prog {
            state.seg_progress = frac;
            state.status = format!("Segmenting… {:.0}% — {}", frac * 100.0, phase);
        }
        let received = state.seg_job.as_ref().unwrap().rx.lock().unwrap().try_recv();
        match received {
            Ok(out) => {
                state.seg_job = None;
                state.seg_active = false;
                finish_segmentation(&mut state, out,
                    &mut commands, &mut meshes, &mut materials);
            }
            Err(TryRecvError::Empty) => {} // still computing; keep the frame alive
            Err(TryRecvError::Disconnected) => {
                state.seg_job = None;
                state.seg_active = false;
                state.status = "Segmentation failed (compute thread dropped)".into();
            }
        }
        return;
    }

    if !state.needs_run { return; }
    state.needs_run = false;

    for e in &existing { commands.entity(e).despawn(); }
    state.detected.clear();
    state.raw_planes.clear();
    state.merged_planes.clear();
    state.building_exterior.clear(); // finish_segmentation repopulates for Building

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

    // Run segmentation off the render thread; the result is consumed on a later
    // frame by the poll block above, so the UI keeps painting the progress bar.
    let sigma_max = (state.dist_thresh * state.sigma_factor) as f64;
    let (k, angle, min_cluster, dist, mode, max_iters, conf) = (
        state.k_neighbors, state.angle_thresh.to_radians(), state.min_cluster_size,
        state.dist_thresh, state.ransac_mode, state.max_iterations, state.confidence,
    );
    let progress = Arc::new(Mutex::new((0.0f32, String::new())));
    let (tx, rx) = mpsc::channel();
    let prog = progress.clone();
    if state.plane_method == PlaneMethod::Building {
        // Full building pipeline: downsamples internally and returns walls with
        // footprint-based exterior labels + the estimated up. The plane indices
        // refer to the returned (downsampled) points, which become all_pts.
        spawn_compute(move || {
            if let Ok(mut g) = prog.lock() {
                *g = (0.5, "Building: align → storeys → walls".into());
            }
            let scene = reconstruct_building(&all_pts, &BuildingParams::default());
            let mut planes = Vec::with_capacity(scene.walls.len());
            let mut exterior = Vec::with_capacity(scene.walls.len());
            for w in &scene.walls {
                planes.push((w.normal, w.d, w.inlier_indices.clone()));
                exterior.push(w.is_exterior);
            }
            let _ = tx.send(SegOut {
                planes,
                all_pts: scene.points,
                building: Some((exterior, scene.up)),
            });
        });
    } else {
        let estimator: Box<dyn PlaneEstimator + Send> = match state.plane_method {
            PlaneMethod::RegionGrowing => Box::new(RegionGrowing {
                k, angle_thresh: angle, min_cluster_size: min_cluster, dist_thresh: dist,
                mode, sigma_max, max_iterations: max_iters, confidence: conf,
            }),
            PlaneMethod::Peeling => Box::new(GlobalPlanePeeling {
                k, normal_consensus: false, dist_thresh: dist, angle_thresh: angle,
                min_support: min_cluster, max_planes: 60, max_iterations: max_iters, confidence: conf,
            }),
            // Manhattan (and any non-building fallback): orientation-vote
            // tolerance widened — the region-growing angle slider (~10°) is far
            // too tight here and leaves whole regions unassigned.
            _ => Box::new(ManhattanPlanes {
                k, dist_thresh: dist,
                angle_thresh: angle.max(35f32.to_radians()),
                min_support: min_cluster,
            }),
        };
        spawn_compute(move || {
            let planes = estimator.estimate_with_progress(&all_pts, &mut |f, phase| {
                if let Ok(mut g) = prog.lock() {
                    *g = (f, phase.to_string());
                }
            });
            let _ = tx.send(SegOut { planes, all_pts, building: None });
        });
    }
    state.seg_job = Some(SegJob { progress, rx: Mutex::new(rx) });
    state.seg_active = true;
    state.seg_progress = 0.0;
    state.status = "Segmenting…".into();
}

/// Estimate the up direction as the dominant plane normal (floors/ceilings have
/// the most support), via the largest eigenvector of Σ count·nnᵀ. Used to tell
/// walls from floors for the dollhouse.
fn estimate_up(planes: &[([f32; 3], f32, Vec<usize>)]) -> [f32; 3] {
    use nalgebra::{Matrix3, SymmetricEigen, Vector3};
    let mut m = Matrix3::<f64>::zeros();
    for (n, _, idx) in planes {
        let v = Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64);
        m += v * v.transpose() * idx.len() as f64;
    }
    if m.trace() < 1e-9 {
        return [0.0, 1.0, 0.0];
    }
    let eig = SymmetricEigen::new(m);
    let mut best = 0;
    for i in 1..3 {
        if eig.eigenvalues[i] > eig.eigenvalues[best] {
            best = i;
        }
    }
    let e = eig.eigenvectors.column(best);
    let v = Vector3::new(e[0], e[1], e[2]).normalize();
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// Turn a finished segmentation into rendered entities and status. Runs on the
/// main thread (touches `Commands`/`Assets`) once the background job delivers.
#[allow(clippy::too_many_arguments)]
fn finish_segmentation(
    state: &mut VoxelPlaneState,
    out: SegOut,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let SegOut { planes, all_pts, building } = out;
    state.raw_planes = planes;
    // Building walls are all vertical, so estimate_up (dominant plane normal)
    // would return a horizontal vector and break the dollhouse wall filter — use
    // the pipeline's up instead. Also set the exterior override; clear it for
    // estimator methods so spawn_planes falls back to the side-test.
    match &building {
        Some((exterior, up)) => {
            state.up_dir = *up;
            state.building_exterior = exterior.clone();
        }
        None => {
            state.up_dir = estimate_up(&state.raw_planes);
            state.building_exterior.clear();
        }
    }
    state.all_pts_cache = all_pts.clone();
    let point_size = state.point_size;

    let raw_planes_ref: &[([f32;3], f32, Vec<usize>)] = &state.raw_planes;
    let mut used = vec![false; all_pts.len()];
    for (_, _, inliers) in raw_planes_ref { for &i in inliers { if i < used.len() { used[i] = true; } } }
    let leftover: Vec<[f32; 3]> = all_pts.iter().enumerate()
        .filter(|&(i, _)| !used[i]).map(|(_, &p)| p).collect();

    let raw_planes_clone: Vec<([f32;3], f32, Vec<usize>)> = state.raw_planes.iter()
        .map(|(n,d,v)| (*n,*d,v.clone())).collect();
    let exterior_thresh = state.dollhouse_exterior_thresh;
    let dist_thresh = state.dist_thresh;
    let color_mode = state.plane_color_mode;
    // Building's all_pts is the downsampled cloud, so full-res loaded_colors
    // wouldn't index-match — force palette by passing no colors.
    let plane_colors = if building.is_some() { Vec::new() } else { state.loaded_colors.clone() };
    let exterior_override = building.as_ref().map(|(e, _)| e.clone());
    let (mut det, mut ents, mut vis) = (vec![], vec![], vec![]);
    spawn_planes(&raw_planes_clone, &all_pts, commands,
        meshes, materials, point_size, exterior_thresh, dist_thresh, color_mode, &plane_colors,
        exterior_override.as_deref(),
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
    color_mode: PlaneColorMode,
    plane_colors: &[[u8; 3]], // original per-point RGB, indexed like all_pts (empty = none)
    // Per-plane exterior labels (Building method's footprint classifier). When
    // Some, overrides classify_plane's side-test; when None, the side-test wins.
    exterior_override: Option<&[bool]>,
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
        // Canonicalize the normal inward and classify exterior via the shared
        // library helper — the same code path `segment_for_dollhouse` runs, so
        // the interactive Segment→Merge→Grow steps stay in sync with the lib.
        // margin excludes the near-plane band (canonicalize_margin_factor = 3.0).
        let (canonical_normal, canonical_d, centroid, side_exterior) =
            classify_plane(*normal, *d, inliers, all_pts, dist_thresh * 3.0, exterior_thresh);
        // Prefer the footprint exterior label when the Building method supplied one.
        let is_exterior = exterior_override
            .and_then(|o| o.get(pi).copied())
            .unwrap_or(side_exterior);

        detected.push((canonical_normal, canonical_d, pts3d.len(), centroid, is_exterior));

        // Original-RGB mode bakes per-point color into the mesh; palette/gray
        // use a flat material color. Fall back to palette if colors are missing.
        let use_rgb = color_mode == PlaneColorMode::Original && !plane_colors.is_empty();
        let (pt_mesh, pt_color) = if use_rgb {
            let cols: Vec<[u8; 3]> = inliers.iter()
                .map(|&i| plane_colors.get(i).copied().unwrap_or([180, 180, 180]))
                .collect();
            (make_cloud_mesh_rgb(&pts3d, &cols, point_size), Color::WHITE)
        } else {
            let c = match color_mode {
                PlaneColorMode::Gray => Color::srgb(0.65, 0.65, 0.65),
                _ => Color::srgb(r, g, b),
            };
            (make_cloud_mesh(&pts3d, point_size), c)
        };
        let pts_entity = if !pts3d.is_empty() {
            Some(commands.spawn((
                Mesh3d(meshes.add(pt_mesh)),
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

// ─── Cloud helpers ────────────────────────────────────────────────────────────

/// Extract RGB colors from a PointCloud. Returns empty vec if no color fields.
impl VoxelPlaneState {
    /// Populate `loaded_pts`/`loaded_colors` from a decoded cloud and mark it
    /// for redraw. Shared by the native file read and the browser fetch path.
    fn set_cloud(&mut self, cloud: &spatialrust_inlier::PointCloud) {
        self.loaded_colors = cloud_rgb(cloud);
        self.loaded_pts = cloud_xyz(cloud);
        let color_info = if self.loaded_colors.is_empty() { "" } else { " (RGB)" };
        self.status =
            format!("Loaded {} pts{color_info} — click Segment to run", self.loaded_pts.len());
        self.detected.clear();
        self.needs_reload = true;
    }
}

/// Kick off a background HTTP fetch of `url` (relative to the page origin) into
/// the shared slot. The browser has no filesystem, so the "Load" button fetches
/// the bundled asset instead of reading a path.
#[cfg(target_arch = "wasm32")]
fn start_fetch(slot: &std::sync::Arc<std::sync::Mutex<Fetch>>, url: &str) {
    *slot.lock().unwrap() = Fetch::Pending;
    let slot = slot.clone();
    ehttp::fetch(ehttp::Request::get(url), move |result| {
        let next = match result {
            Ok(resp) if resp.ok => Fetch::Done(resp.bytes),
            Ok(resp) => Fetch::Failed(format!("HTTP {} {}", resp.status, resp.status_text)),
            Err(e) => Fetch::Failed(e),
        };
        *slot.lock().unwrap() = next;
    });
}

/// Drain a completed fetch: parse the PLY bytes in-memory and load the cloud,
/// or surface the error in the status line. No-op while idle/pending.
#[cfg(target_arch = "wasm32")]
fn poll_fetch(state: &mut VoxelPlaneState) {
    let done = {
        let mut slot = state.fetch.lock().unwrap();
        matches!(&*slot, Fetch::Done(_) | Fetch::Failed(_)).then(|| std::mem::take(&mut *slot))
    };
    match done {
        Some(Fetch::Done(bytes)) => {
            match spatialrust_inlier::io::read_ply(&mut std::io::Cursor::new(bytes)) {
                Ok(cloud) => state.set_cloud(&cloud),
                Err(e) => state.status = format!("Parse error: {e}"),
            }
        }
        Some(Fetch::Failed(e)) => state.status = format!("Fetch error: {e}"),
        _ => {}
    }
}

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
    let up = bevy::math::Vec3::from(state.up_dir);
    // Building mode returns floor+ceiling as confident structure, and its
    // canonical inward normals make the camera-facing test hide the ceiling
    // (camera above) while keeping the floor visible — so DON'T skip horizontals
    // there. Other methods still keep floors/ceilings visible for legibility.
    let building = !state.building_exterior.is_empty();

    for (i, &(normal, _, _, centroid, is_exterior)) in state.detected.iter().enumerate() {
        // Non-building: only hide roughly-vertical walls (|normal·up| ~0).
        if !building && bevy::math::Vec3::from(normal).dot(up).abs() > 0.5 { continue; }
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
