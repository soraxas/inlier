/// Region-growing + per-cluster RANSAC plane segmentation (Ling et al., ISPRS 2024).
///
/// Step 1: Compute per-point PCA normal + curvature via k-NN.
/// Step 2: Region growing sorted by curvature (flattest seed first), angle threshold.
/// Step 3: RANSAC on each grown cluster, then sweep remaining points with
///         distance + normal constraints to recover edge points.
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use std::collections::{HashMap, VecDeque};

use crate::AppDemo;
use crate::plane_demo::{make_cloud_mesh, make_alpha_shape_mesh};
use spatialrust_inlier::{fit_plane_msac, fit_plane_magsac_raw};

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
            .add_systems(Update, voxel_plane_scene.run_if(in_state(AppDemo::VoxelPlane)));
    }
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
    /// sigma_max for MAGSAC (multiples of dist_thresh; UI shows the multiplier)
    pub sigma_factor: f32,
    // runtime
    pub needs_run: bool,
    pub status: String,
    pub detected: Vec<([f32; 3], f32, usize)>,
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
            needs_run: true,
            status: String::new(),
            detected: vec![],
        }
    }
}

#[derive(Component)]
struct VpEntity;

fn on_enter(mut s: ResMut<VoxelPlaneState>) { s.needs_run = true; }

fn on_exit(mut commands: Commands, q: Query<Entity, With<VpEntity>>) {
    for e in &q { commands.entity(e).despawn(); }
}

fn voxel_plane_ui(mut contexts: EguiContexts, mut state: ResMut<VoxelPlaneState>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Panel::left("vp_panel").default_size(260.0).show(ctx, |ui| {
        ui.heading("Region-Growing + RANSAC");
        ui.small("Ling et al. ISPRS 2024");
        ui.separator();

        egui::CollapsingHeader::new("Generation").default_open(true).show(ui, |ui| {
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
        if ui.button("Generate & Segment").clicked() {
            if state.auto_seed { state.seed = lcg_next(state.seed); }
            state.needs_run = true;
        }

        if !state.status.is_empty() {
            ui.separator();
            ui.small(&state.status);
        }

        if !state.detected.is_empty() {
            ui.separator();
            ui.colored_label(egui::Color32::LIGHT_BLUE,
                format!("Detected {} planes", state.detected.len()));
            for (i, &(n, d, cnt)) in state.detected.iter().enumerate() {
                let [r, g, b] = PT_COLORS[i % MAX_PLANES];
                ui.colored_label(
                    egui::Color32::from_rgb((r*255.) as u8, (g*255.) as u8, (b*255.) as u8),
                    format!("Plane {} | n=[{:.2},{:.2},{:.2}] d={:.3} | {} pts",
                        i+1, n[0], n[1], n[2], d, cnt),
                );
            }
        }
    });
}

fn voxel_plane_scene(
    mut commands: Commands,
    mut state: ResMut<VoxelPlaneState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<VpEntity>>,
) {
    if !state.needs_run { return; }
    state.needs_run = false;

    for e in &existing { commands.entity(e).despawn(); }
    state.detected.clear();

    let all_pts = synthetic_multi_plane(
        state.n_planes, state.n_inliers, state.noise_std, state.n_outliers, state.seed,
    );

    let sigma_max = (state.dist_thresh * state.sigma_factor) as f64;
    let planes = region_growing_ransac(
        &all_pts,
        state.k_neighbors,
        state.angle_thresh.to_radians(),
        state.min_cluster_size,
        state.dist_thresh,
        state.ransac_mode,
        sigma_max,
    );

    let mut used = vec![false; all_pts.len()];

    for (pi, (normal, d, inliers)) in planes.iter().enumerate() {
        let ci = pi % MAX_PLANES;
        let [r, g, b] = PT_COLORS[ci];
        let [mr, mg, mb, ma] = MESH_COLORS[ci];

        for &i in inliers { if i < used.len() { used[i] = true; } }
        let pts3d: Vec<[f32; 3]> = inliers.iter().map(|&i| all_pts[i]).collect();

        state.detected.push((*normal, *d, pts3d.len()));

        if !pts3d.is_empty() {
            commands.spawn((
                Mesh3d(meshes.add(make_cloud_mesh(&pts3d, state.point_size))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(r, g, b),
                    unlit: true,
                    ..default()
                })),
                Transform::default(),
                VpEntity,
            ));
        }

        if let Some(mesh) = make_alpha_shape_mesh(*normal, *d, &pts3d) {
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
            ));
        }
    }

    let leftover: Vec<[f32; 3]> = all_pts.iter().enumerate()
        .filter(|&(i, _)| !used[i]).map(|(_, &p)| p).collect();
    if !leftover.is_empty() {
        commands.spawn((
            Mesh3d(meshes.add(make_cloud_mesh(&leftover, state.point_size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.45, 0.45, 0.45),
                unlit: true,
                ..default()
            })),
            Transform::default(),
            VpEntity,
        ));
    }

    state.status = format!(
        "{} pts | {} planes | {} unassigned",
        all_pts.len(), planes.len(), leftover.len()
    );
}

// ─── Core pipeline ────────────────────────────────────────────────────────────

fn region_growing_ransac(
    pts: &[[f32; 3]],
    k: usize,
    angle_thresh: f32,
    min_cluster_size: usize,
    dist_thresh: f32,
    mode: RansacMode,
    sigma_max: f64,
) -> Vec<([f32; 3], f32, Vec<usize>)> {
    let n = pts.len();
    if n < 3 { return vec![]; }

    // Build spatial grid for KNN queries.
    // Cell size = median NN distance estimate: sample 200 pts, compute mean NN, use that.
    let cell_size = estimate_cell_size(pts);
    let grid = build_grid(pts, cell_size);

    // Step 1: per-point normals and curvatures.
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

    // Step 2: region growing sorted by curvature ascending (flattest = best seed).
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
                // Normal agreement check.
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

    // Step 3: RANSAC per cluster + sweep remaining unassigned points.
    let mut assigned = vec![false; n];
    let mut result: Vec<([f32; 3], f32, Vec<usize>)> = Vec::new();

    for cluster in clusters {
        let unassigned_count = cluster.iter().filter(|&&i| !assigned[i]).count();
        if unassigned_count < min_cluster_size { continue; }

        let unassigned: Vec<usize> = cluster.iter().cloned().filter(|&i| !assigned[i]).collect();
        let cluster_pts: Vec<[f32;3]> = unassigned.iter().map(|&i| pts[i]).collect();

        let fit = match mode {
            RansacMode::Simple => ransac_plane(&cluster_pts, dist_thresh, 200, 42)
                .map(|(n, d)| (n, d)),
            RansacMode::Msac => fit_plane_msac(&cluster_pts, dist_thresh as f64, None)
                .map(|(n, d, _)| (n, d)),
            RansacMode::Magsac => fit_plane_magsac_raw(&cluster_pts, sigma_max, None)
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

    result
}

/// Minimal RANSAC plane fitter operating on raw [f32;3] slices.
/// Returns (unit_normal, d) such that normal·p + d ≈ 0, or None.
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

// ─── Spatial helpers ──────────────────────────────────────────────────────────

fn estimate_cell_size(pts: &[[f32; 3]]) -> f32 {
    let n = pts.len();
    let stride = (n / 200).max(1);
    let sample: Vec<[f32; 3]> = pts.iter().step_by(stride).cloned().collect();
    let k = sample.len();
    if k < 2 { return 0.1; }
    let mut sum = 0.0f32;
    for i in 0..k {
        let mut best = f32::MAX;
        for j in 0..k {
            if i == j { continue; }
            let d = dist2(sample[i], sample[j]).sqrt();
            if d < best { best = d; }
        }
        sum += best;
    }
    // Use 2× mean NN as cell size so neighbors fit within 1-ring.
    ((sum / k as f32) * 2.0).max(1e-5)
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

/// Return indices of k nearest neighbors of pts[idx] using the spatial grid.
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

    // Search radius: expand rings until we have enough candidates.
    let mut candidates: Vec<(f32, usize)> = Vec::new();
    'outer: for ring in 1i32..=5 {
        for dx in -ring..=ring {
            for dy in -ring..=ring {
                for dz in -ring..=ring {
                    // Only process the shell of this ring.
                    if dx.abs() < ring && dy.abs() < ring && dz.abs() < ring { continue; }
                    if let Some(cell) = grid.get(&(cx+dx, cy+dy, cz+dz)) {
                        for &j in cell {
                            if j == idx { continue; }
                            candidates.push((dist2(p, pts[j]), j));
                        }
                    }
                }
            }
        }
        if candidates.len() >= k { break 'outer; }
    }
    // Also include same cell.
    if let Some(cell) = grid.get(&(cx, cy, cz)) {
        for &j in cell {
            if j != idx { candidates.push((dist2(p, pts[j]), j)); }
        }
    }

    candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    candidates.truncate(k);
    candidates.into_iter().map(|(_, j)| j).collect()
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)
}

// ─── PCA ──────────────────────────────────────────────────────────────────────

/// Returns (unit_normal, curvature) where curvature = λ₀ / (λ₀ + λ₁ + λ₂).
/// Uses the shift trick for the normal and the Rayleigh quotient for curvature.
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
        let planes = region_growing_ransac(&pts, 20, 10f32.to_radians(), 30, 0.08, RansacMode::Msac, 0.0);
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
        let planes = region_growing_ransac(&pts, 20, 10f32.to_radians(), 30, 0.08, RansacMode::Magsac, sigma_max);
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
