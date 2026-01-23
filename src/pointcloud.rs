//! Point cloud registration utilities and colored ICP pipeline.
use std::collections::HashMap;

use nalgebra::{Matrix3, Matrix4, SMatrix, SVD, SVector, SymmetricEigen, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::core::Estimator;
use crate::estimators::RigidTransformEstimator;
use crate::features::{FeatureMatchSettings, FpfhSettings, compute_fpfh_features, match_features};
use crate::models::RigidTransform;
use crate::robust::RobustLoss;
use crate::samplers::{NeighborhoodGraph, UsearchNeighborhoodGraph};
use crate::types::DataMatrix;

type Correspondence = (usize, usize);

#[derive(Clone, Debug)]
pub struct PointCloud {
    pub points: DataMatrix,
    pub colors: Option<DataMatrix>,
    pub normals: Option<DataMatrix>,
    pub color_gradients: Option<DataMatrix>,
}

impl PointCloud {
    pub fn new(points: DataMatrix, colors: Option<DataMatrix>) -> Result<Self, String> {
        if points.ncols() != 3 {
            return Err("points must be Nx3 matrix".to_string());
        }
        if let Some(colors) = &colors
            && (colors.ncols() != 3 || colors.nrows() != points.nrows())
        {
            return Err("colors must be Nx3 matrix aligned to points".to_string());
        }
        Ok(Self {
            points,
            colors,
            normals: None,
            color_gradients: None,
        })
    }

    pub fn from_xyzrgb(data: &DataMatrix) -> Result<Self, String> {
        if data.ncols() != 6 {
            return Err("data must be Nx6 matrix (xyzrgb)".to_string());
        }
        let n = data.nrows();
        let mut points = DataMatrix::zeros(n, 3);
        let mut colors = DataMatrix::zeros(n, 3);
        for i in 0..n {
            points[(i, 0)] = data[(i, 0)];
            points[(i, 1)] = data[(i, 1)];
            points[(i, 2)] = data[(i, 2)];
            colors[(i, 0)] = data[(i, 3)];
            colors[(i, 1)] = data[(i, 4)];
            colors[(i, 2)] = data[(i, 5)];
        }
        Self::new(points, Some(colors))
    }

    pub fn from_xyz(data: &DataMatrix) -> Result<Self, String> {
        if data.ncols() != 3 {
            return Err("data must be Nx3 matrix".to_string());
        }
        Self::new(data.clone(), None)
    }

    pub fn len(&self) -> usize {
        self.points.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.points.nrows() == 0
    }

    fn point(&self, idx: usize) -> Vector3<f64> {
        Vector3::new(
            self.points[(idx, 0)],
            self.points[(idx, 1)],
            self.points[(idx, 2)],
        )
    }

    fn color(&self, idx: usize) -> Option<Vector3<f64>> {
        self.colors
            .as_ref()
            .map(|c| Vector3::new(c[(idx, 0)], c[(idx, 1)], c[(idx, 2)]))
    }

    fn normal(&self, idx: usize) -> Option<Vector3<f64>> {
        self.normals
            .as_ref()
            .map(|n| Vector3::new(n[(idx, 0)], n[(idx, 1)], n[(idx, 2)]))
    }

    fn color_gradient(&self, idx: usize) -> Option<Vector3<f64>> {
        self.color_gradients
            .as_ref()
            .map(|g| Vector3::new(g[(idx, 0)], g[(idx, 1)], g[(idx, 2)]))
    }

    pub fn voxel_downsample(&self, voxel_size: f64) -> Result<Self, String> {
        if voxel_size <= 0.0 {
            return Err("voxel_size must be > 0".to_string());
        }
        if self.is_empty() {
            return Ok(self.clone());
        }

        #[derive(Clone, Copy)]
        struct Accumulator {
            sum: Vector3<f64>,
            color_sum: Vector3<f64>,
            count: usize,
        }

        let mut map: HashMap<(i64, i64, i64), Accumulator> = HashMap::new();
        let use_colors = self.colors.is_some();

        for i in 0..self.len() {
            let p = self.point(i);
            let key = (
                (p.x / voxel_size).floor() as i64,
                (p.y / voxel_size).floor() as i64,
                (p.z / voxel_size).floor() as i64,
            );
            let entry = map.entry(key).or_insert(Accumulator {
                sum: Vector3::zeros(),
                color_sum: Vector3::zeros(),
                count: 0,
            });
            entry.sum += p;
            if use_colors && let Some(c) = self.color(i) {
                entry.color_sum += c;
            }
            entry.count += 1;
        }

        let mut points = Vec::with_capacity(map.len() * 3);
        let mut colors = Vec::with_capacity(map.len() * 3);
        for acc in map.values() {
            let inv = 1.0 / acc.count as f64;
            let p = acc.sum * inv;
            points.extend_from_slice(&[p.x, p.y, p.z]);
            if use_colors {
                let c = acc.color_sum * inv;
                colors.extend_from_slice(&[c.x, c.y, c.z]);
            }
        }

        let points = DataMatrix::from_row_slice(points.len() / 3, 3, &points);
        let colors = if use_colors {
            Some(DataMatrix::from_row_slice(colors.len() / 3, 3, &colors))
        } else {
            None
        };

        Self::new(points, colors)
    }
}

pub fn preprocess_point_cloud(
    cloud: &PointCloud,
    settings: &PreprocessSettings,
) -> Result<(PointCloud, DataMatrix), String> {
    let down = cloud.voxel_downsample(settings.voxel_size)?;
    let normal_estimator = KnnNormalEstimator;
    let down = normal_estimator.estimate(&down, settings.normal_radius, settings.normal_max_nn)?;
    let fpfh_settings = FpfhSettings {
        radius: settings.fpfh_radius,
        max_nn: settings.fpfh_max_nn,
    };
    let fpfh = compute_fpfh_features(&down, &fpfh_settings)?;
    Ok((down, fpfh))
}

pub trait DownsampleStrategy {
    fn downsample(&self, cloud: &PointCloud, voxel_size: f64) -> Result<PointCloud, String>;
}

pub struct VoxelDownsample;

impl DownsampleStrategy for VoxelDownsample {
    fn downsample(&self, cloud: &PointCloud, voxel_size: f64) -> Result<PointCloud, String> {
        cloud.voxel_downsample(voxel_size)
    }
}

pub trait NormalEstimationStrategy {
    fn estimate(
        &self,
        cloud: &PointCloud,
        radius: f64,
        max_nn: usize,
    ) -> Result<PointCloud, String>;
}

pub struct KnnNormalEstimator;

impl KnnNormalEstimator {
    fn build_index(points: &DataMatrix) -> Result<Index, String> {
        let options = IndexOptions {
            dimensions: 3,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options).map_err(|_| "failed to build index".to_string())?;
        index
            .reserve(points.nrows())
            .map_err(|_| "failed to reserve index".to_string())?;
        for i in 0..points.nrows() {
            let vec = [
                points[(i, 0)] as f32,
                points[(i, 1)] as f32,
                points[(i, 2)] as f32,
            ];
            index
                .add(i as u64, &vec)
                .map_err(|_| "failed to add point to index".to_string())?;
        }
        Ok(index)
    }

    fn neighbors_within(
        index: &Index,
        points: &DataMatrix,
        center: usize,
        radius: f64,
        max_nn: usize,
    ) -> Vec<usize> {
        let query = [
            points[(center, 0)] as f32,
            points[(center, 1)] as f32,
            points[(center, 2)] as f32,
        ];
        let k = (max_nn + 1).min(points.nrows());
        let mut out = Vec::new();
        let radius_sq = radius * radius;
        if let Ok(results) = index.search(&query, k) {
            for (key, dist) in results.keys.iter().zip(results.distances.iter()) {
                let idx = *key as usize;
                if idx == center {
                    continue;
                }
                if (*dist as f64) <= radius_sq {
                    out.push(idx);
                }
            }
        }
        out
    }
}

impl NormalEstimationStrategy for KnnNormalEstimator {
    fn estimate(
        &self,
        cloud: &PointCloud,
        radius: f64,
        max_nn: usize,
    ) -> Result<PointCloud, String> {
        if cloud.is_empty() {
            return Ok(cloud.clone());
        }
        let index = Self::build_index(&cloud.points)?;
        let mut normals = DataMatrix::zeros(cloud.len(), 3);
        for i in 0..cloud.len() {
            let neighbors = Self::neighbors_within(&index, &cloud.points, i, radius, max_nn);
            if neighbors.len() < 3 {
                continue;
            }
            let mut mean = Vector3::zeros();
            for &j in &neighbors {
                mean += cloud.point(j);
            }
            mean /= neighbors.len() as f64;

            let mut cov = Matrix3::zeros();
            for &j in &neighbors {
                let d = cloud.point(j) - mean;
                cov += d * d.transpose();
            }
            let eig = SymmetricEigen::new(cov);
            let mut min_idx = 0;
            let mut min_val = eig.eigenvalues[0];
            for k in 1..3 {
                if eig.eigenvalues[k] < min_val {
                    min_val = eig.eigenvalues[k];
                    min_idx = k;
                }
            }
            let n = eig.eigenvectors.column(min_idx).normalize();
            normals[(i, 0)] = n.x;
            normals[(i, 1)] = n.y;
            normals[(i, 2)] = n.z;
        }

        let mut out = cloud.clone();
        out.normals = Some(normals);
        Ok(out)
    }
}

pub trait ColorGradientEstimationStrategy {
    fn estimate(
        &self,
        cloud: &PointCloud,
        radius: f64,
        max_nn: usize,
    ) -> Result<PointCloud, String>;
}

pub struct KnnColorGradientEstimator;

impl ColorGradientEstimationStrategy for KnnColorGradientEstimator {
    fn estimate(
        &self,
        cloud: &PointCloud,
        radius: f64,
        max_nn: usize,
    ) -> Result<PointCloud, String> {
        let colors = match &cloud.colors {
            Some(c) => c,
            None => return Ok(cloud.clone()),
        };
        let normals = match &cloud.normals {
            Some(n) => n,
            None => return Ok(cloud.clone()),
        };
        if cloud.is_empty() {
            return Ok(cloud.clone());
        }
        let index = KnnNormalEstimator::build_index(&cloud.points)?;
        let mut grads = DataMatrix::zeros(cloud.len(), 3);

        for i in 0..cloud.len() {
            let n = Vector3::new(normals[(i, 0)], normals[(i, 1)], normals[(i, 2)]);
            if n.norm() < 1e-9 {
                continue;
            }
            let tangent = if n.z.abs() < 0.9 {
                Vector3::z()
            } else {
                Vector3::x()
            };
            let u = n.cross(&tangent).normalize();
            let v = n.cross(&u).normalize();

            let neighbors =
                KnnNormalEstimator::neighbors_within(&index, &cloud.points, i, radius, max_nn);
            if neighbors.len() < 3 {
                continue;
            }

            let i0 = (colors[(i, 0)] + colors[(i, 1)] + colors[(i, 2)]) / 3.0;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            let mut syy = 0.0;
            let mut sxi = 0.0;
            let mut syi = 0.0;
            let center = cloud.point(i);

            for &j in &neighbors {
                let pj = cloud.point(j);
                let d = pj - center;
                let x = d.dot(&u);
                let y = d.dot(&v);
                let ij = (colors[(j, 0)] + colors[(j, 1)] + colors[(j, 2)]) / 3.0;
                let di = ij - i0;
                sxx += x * x;
                sxy += x * y;
                syy += y * y;
                sxi += x * di;
                syi += y * di;
            }

            let det = sxx * syy - sxy * sxy;
            if det.abs() < 1e-12 {
                continue;
            }
            let inv_det = 1.0 / det;
            let a = (syy * sxi - sxy * syi) * inv_det;
            let b = (sxx * syi - sxy * sxi) * inv_det;

            let grad = u * a + v * b;
            grads[(i, 0)] = grad.x;
            grads[(i, 1)] = grad.y;
            grads[(i, 2)] = grad.z;
        }

        let mut out = cloud.clone();
        out.color_gradients = Some(grads);
        Ok(out)
    }
}

pub trait CorrespondenceStrategy {
    fn prepare(&mut self, target: &PointCloud) -> Result<(), String>;
    fn nearest(&self, point: &Vector3<f64>) -> Option<(usize, f64)>;
}

pub struct UsearchCorrespondence {
    index: Option<Index>,
    target: Option<PointCloud>,
}

impl Default for UsearchCorrespondence {
    fn default() -> Self {
        Self::new()
    }
}

impl UsearchCorrespondence {
    pub fn new() -> Self {
        Self {
            index: None,
            target: None,
        }
    }
}

impl CorrespondenceStrategy for UsearchCorrespondence {
    fn prepare(&mut self, target: &PointCloud) -> Result<(), String> {
        let index = KnnNormalEstimator::build_index(&target.points)?;
        self.index = Some(index);
        self.target = Some(target.clone());
        Ok(())
    }

    fn nearest(&self, point: &Vector3<f64>) -> Option<(usize, f64)> {
        let index = self.index.as_ref()?;
        let query = [point.x as f32, point.y as f32, point.z as f32];
        let results = index.search(&query, 1).ok()?;
        let key = *results.keys.first()? as usize;
        let dist = *results.distances.first()? as f64;
        Some((key, dist))
    }
}

#[derive(Clone, Debug)]
pub struct ColoredIcpSettings {
    pub distance_threshold: f64,
    pub lambda_geometric: f64,
    pub lambda_color: f64,
    pub normal_radius: f64,
    pub normal_max_nn: usize,
    pub gradient_radius: f64,
    pub gradient_max_nn: usize,
    pub robust_loss: RobustLoss,
}

impl Default for ColoredIcpSettings {
    fn default() -> Self {
        Self {
            distance_threshold: 0.0,
            lambda_geometric: 0.968,
            lambda_color: 1.0 - 0.968,
            normal_radius: 0.04,
            normal_max_nn: 30,
            gradient_radius: 0.04,
            gradient_max_nn: 30,
            robust_loss: RobustLoss::None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IcpConvergenceCriteria {
    pub relative_fitness: f64,
    pub relative_rmse: f64,
    pub max_iteration: usize,
}

impl Default for IcpConvergenceCriteria {
    fn default() -> Self {
        Self {
            relative_fitness: 1e-6,
            relative_rmse: 1e-6,
            max_iteration: 50,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColoredIcpScale {
    pub voxel_radius: f64,
    pub max_iteration: usize,
}

#[derive(Clone, Debug)]
pub struct RegistrationResult {
    pub transformation: Matrix4<f64>,
    pub fitness: f64,
    pub inlier_rmse: f64,
    pub correspondence_count: usize,
    pub iterations: usize,
}

#[derive(Clone, Debug)]
pub struct PreprocessSettings {
    pub voxel_size: f64,
    pub normal_radius: f64,
    pub normal_max_nn: usize,
    pub fpfh_radius: f64,
    pub fpfh_max_nn: usize,
}

impl Default for PreprocessSettings {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            normal_radius: 0.1,
            normal_max_nn: 30,
            fpfh_radius: 0.25,
            fpfh_max_nn: 100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GlobalRegistrationSettings {
    pub distance_threshold: f64,
    pub ransac_n: usize,
    pub max_iterations: usize,
    pub confidence: f64,
    pub mutual_filter: bool,
    pub max_correspondences: usize,
    pub seed: u64,
    pub checkers: Vec<CorrespondenceChecker>,
}

impl Default for GlobalRegistrationSettings {
    fn default() -> Self {
        let distance_threshold = 0.075;
        Self {
            distance_threshold,
            ransac_n: 3,
            max_iterations: 100_000,
            confidence: 0.999,
            mutual_filter: true,
            max_correspondences: 0,
            seed: 42,
            checkers: vec![
                CorrespondenceChecker::EdgeLength { threshold: 0.9 },
                CorrespondenceChecker::Distance {
                    threshold: distance_threshold,
                },
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub enum CorrespondenceChecker {
    EdgeLength { threshold: f64 },
    Distance { threshold: f64 },
    Normal { max_angle: f64 },
}

impl CorrespondenceChecker {
    fn check_sample(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        correspondences: &[Correspondence],
        transform: &Matrix4<f64>,
    ) -> bool {
        match *self {
            CorrespondenceChecker::EdgeLength { threshold } => {
                edge_length_check(source, target, correspondences, threshold)
            }
            CorrespondenceChecker::Distance { threshold } => {
                distance_check(source, target, correspondences, transform, threshold)
            }
            CorrespondenceChecker::Normal { max_angle } => {
                normal_set_check(source, target, correspondences, max_angle.cos())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct FastGlobalRegistrationOptions {
    pub maximum_correspondence_distance: f64,
    pub max_iterations: usize,
    pub mu: f64,
    pub mu_decay: f64,
}

impl Default for FastGlobalRegistrationOptions {
    fn default() -> Self {
        Self {
            maximum_correspondence_distance: 0.025,
            max_iterations: 64,
            mu: 1.0,
            mu_decay: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub enum GlobalRegistrationMethod {
    Ransac,
    FastGlobalRegistration,
}

pub trait IcpKernel {
    fn solve(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        correspondences: &[Correspondence],
        transform: &Matrix4<f64>,
        settings: &ColoredIcpSettings,
    ) -> Option<Matrix4<f64>>;
}

pub struct ColoredPointToPlaneKernel;

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn se3_exp(delta: &SVector<f64, 6>) -> Matrix4<f64> {
    let w = Vector3::new(delta[0], delta[1], delta[2]);
    let v = Vector3::new(delta[3], delta[4], delta[5]);
    let theta = w.norm();
    let (r, v_mat) = if theta > 1e-12 {
        let w_hat = skew(&w);
        let w_hat2 = w_hat * w_hat;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let r = Matrix3::identity()
            + (sin_t / theta) * w_hat
            + ((1.0 - cos_t) / (theta * theta)) * w_hat2;
        let v_mat = Matrix3::identity()
            + ((1.0 - cos_t) / (theta * theta)) * w_hat
            + ((theta - sin_t) / (theta * theta * theta)) * w_hat2;
        (r, v_mat)
    } else {
        let w_hat = skew(&w);
        let r = Matrix3::identity() + w_hat;
        let v_mat = Matrix3::identity() + 0.5 * w_hat;
        (r, v_mat)
    };
    let t = v_mat * v;
    let mut out = Matrix4::identity();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    out[(0, 3)] = t.x;
    out[(1, 3)] = t.y;
    out[(2, 3)] = t.z;
    out
}

impl ColoredPointToPlaneKernel {
    fn accumulate(
        ata: &mut SMatrix<f64, 6, 6>,
        atb: &mut SVector<f64, 6>,
        j: &SVector<f64, 6>,
        residual: f64,
        weight: f64,
    ) {
        if weight <= 0.0 {
            return;
        }
        *ata += weight * (j * j.transpose());
        *atb += weight * (j * (-residual));
    }
}

impl IcpKernel for ColoredPointToPlaneKernel {
    fn solve(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        correspondences: &[(usize, usize)],
        transform: &Matrix4<f64>,
        settings: &ColoredIcpSettings,
    ) -> Option<Matrix4<f64>> {
        let normals = target.normals.as_ref()?;
        let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);

        let mut ata = SMatrix::<f64, 6, 6>::zeros();
        let mut atb = SVector::<f64, 6>::zeros();

        let use_color = settings.lambda_color > 0.0
            && source.colors.is_some()
            && target.colors.is_some()
            && target.color_gradients.is_some();

        for &(si, ti) in correspondences {
            let p = source.point(si);
            let p_trans = r * p + t;
            let q = Vector3::new(
                target.points[(ti, 0)],
                target.points[(ti, 1)],
                target.points[(ti, 2)],
            );
            let n = Vector3::new(normals[(ti, 0)], normals[(ti, 1)], normals[(ti, 2)]);
            if n.norm() < 1e-9 {
                continue;
            }

            let r_g = n.dot(&(p_trans - q));
            let j_rot = p_trans.cross(&n);
            let j = SVector::<f64, 6>::from_row_slice(&[j_rot.x, j_rot.y, j_rot.z, n.x, n.y, n.z]);
            let w = settings.lambda_geometric * settings.robust_loss.weight(r_g);
            ColoredPointToPlaneKernel::accumulate(&mut ata, &mut atb, &j, r_g, w);

            if use_color {
                let g = target.color_gradient(ti).unwrap_or_else(Vector3::zeros);
                if g.norm() > 0.0 {
                    let cs = source.color(si).unwrap();
                    let ct = target.color(ti).unwrap();
                    let is = (cs.x + cs.y + cs.z) / 3.0;
                    let it = (ct.x + ct.y + ct.z) / 3.0;
                    let r_c = (is - it) - g.dot(&(p_trans - q));
                    let j_rot_c = p_trans.cross(&g);
                    let j_c = SVector::<f64, 6>::from_row_slice(&[
                        j_rot_c.x, j_rot_c.y, j_rot_c.z, g.x, g.y, g.z,
                    ]);
                    let wc = settings.lambda_color * settings.robust_loss.weight(r_c);
                    ColoredPointToPlaneKernel::accumulate(&mut ata, &mut atb, &j_c, r_c, wc);
                }
            }
        }

        let delta = ata.lu().solve(&atb)?;
        Some(se3_exp(&delta))
    }
}

pub struct PointToPlaneKernel;

impl IcpKernel for PointToPlaneKernel {
    fn solve(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        correspondences: &[(usize, usize)],
        transform: &Matrix4<f64>,
        settings: &ColoredIcpSettings,
    ) -> Option<Matrix4<f64>> {
        let normals = target.normals.as_ref()?;
        let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);

        let mut ata = SMatrix::<f64, 6, 6>::zeros();
        let mut atb = SVector::<f64, 6>::zeros();

        for &(si, ti) in correspondences {
            let p = source.point(si);
            let p_trans = r * p + t;
            let q = Vector3::new(
                target.points[(ti, 0)],
                target.points[(ti, 1)],
                target.points[(ti, 2)],
            );
            let n = Vector3::new(normals[(ti, 0)], normals[(ti, 1)], normals[(ti, 2)]);
            if n.norm() < 1e-9 {
                continue;
            }

            let r_g = n.dot(&(p_trans - q));
            let j_rot = p_trans.cross(&n);
            let j = SVector::<f64, 6>::from_row_slice(&[j_rot.x, j_rot.y, j_rot.z, n.x, n.y, n.z]);
            let w = settings.lambda_geometric * settings.robust_loss.weight(r_g);
            if w <= 0.0 {
                continue;
            }
            ata += w * (j * j.transpose());
            atb += w * (j * (-r_g));
        }

        let delta = ata.lu().solve(&atb)?;
        Some(se3_exp(&delta))
    }
}

fn build_correspondence_matrix(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[(usize, usize)],
) -> DataMatrix {
    let mut data = DataMatrix::zeros(correspondences.len(), 6);
    for (i, &(si, ti)) in correspondences.iter().enumerate() {
        data[(i, 0)] = source.points[(si, 0)];
        data[(i, 1)] = source.points[(si, 1)];
        data[(i, 2)] = source.points[(si, 2)];
        data[(i, 3)] = target.points[(ti, 0)];
        data[(i, 4)] = target.points[(ti, 1)];
        data[(i, 5)] = target.points[(ti, 2)];
    }
    data
}

fn estimate_rigid_from_sample(data: &DataMatrix, sample: &[usize]) -> Option<RigidTransform> {
    let estimator = RigidTransformEstimator::new();
    let models = estimator.estimate_model(data, sample);
    models.into_iter().next()
}

fn edge_length_check(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[(usize, usize)],
    threshold: f64,
) -> bool {
    if threshold <= 0.0 || correspondences.len() < 2 {
        return true;
    }
    for i in 0..correspondences.len() {
        for j in (i + 1)..correspondences.len() {
            let (s_i, t_i) = correspondences[i];
            let (s_j, t_j) = correspondences[j];
            let ds = (source.point(s_i) - source.point(s_j)).norm();
            let dt = (target.point(t_i) - target.point(t_j)).norm();
            if ds < 1e-12 || dt < 1e-12 {
                return false;
            }
            let ratio = (ds / dt).min(dt / ds);
            if ratio < threshold {
                return false;
            }
        }
    }
    true
}

fn normal_set_check(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[(usize, usize)],
    cos_threshold: f64,
) -> bool {
    if cos_threshold <= -1.0 {
        return true;
    }
    for &(s_i, t_i) in correspondences {
        let ns = match source.normal(s_i) {
            Some(n) => n,
            None => return true,
        };
        let nt = match target.normal(t_i) {
            Some(n) => n,
            None => return true,
        };
        if ns.norm() < 1e-9 || nt.norm() < 1e-9 {
            return false;
        }
        if ns.normalize().dot(&nt.normalize()) < cos_threshold {
            return false;
        }
    }
    true
}

fn distance_check(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[(usize, usize)],
    transform: &Matrix4<f64>,
    threshold: f64,
) -> bool {
    if threshold <= 0.0 {
        return true;
    }
    let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
    let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
    for &(s_i, t_i) in correspondences {
        let p = source.point(s_i);
        let q = target.point(t_i);
        let p_trans = r * p + t;
        if (p_trans - q).norm() > threshold {
            return false;
        }
    }
    true
}

fn normal_pair_check(
    source: &PointCloud,
    target: &PointCloud,
    correspondence: (usize, usize),
    cos_threshold: f64,
) -> bool {
    if cos_threshold <= -1.0 {
        return true;
    }
    let (s_i, t_i) = correspondence;
    let ns = match source.normal(s_i) {
        Some(n) => n,
        None => return true,
    };
    let nt = match target.normal(t_i) {
        Some(n) => n,
        None => return true,
    };
    if ns.norm() < 1e-9 || nt.norm() < 1e-9 {
        return false;
    }
    ns.normalize().dot(&nt.normalize()) >= cos_threshold
}

fn normal_cos_threshold(checkers: &[CorrespondenceChecker]) -> Option<f64> {
    let mut out = None;
    for checker in checkers {
        if let CorrespondenceChecker::Normal { max_angle } = *checker {
            let cos = max_angle.cos();
            out = Some(out.map(|prev: f64| prev.min(cos)).unwrap_or(cos));
        }
    }
    out
}

fn normalized_checkers(
    checkers: &[CorrespondenceChecker],
    distance_threshold: f64,
) -> Vec<CorrespondenceChecker> {
    checkers
        .iter()
        .map(|checker| match *checker {
            CorrespondenceChecker::Distance { threshold } if threshold <= 0.0 => {
                CorrespondenceChecker::Distance {
                    threshold: distance_threshold,
                }
            }
            _ => checker.clone(),
        })
        .collect()
}

pub fn registration_ransac_based_on_feature_matching(
    source_down: &PointCloud,
    target_down: &PointCloud,
    source_fpfh: &DataMatrix,
    target_fpfh: &DataMatrix,
    settings: &GlobalRegistrationSettings,
) -> Result<RegistrationResult, String> {
    if source_down.is_empty() || target_down.is_empty() {
        return Err("empty point cloud".to_string());
    }

    let match_settings = FeatureMatchSettings {
        mutual: settings.mutual_filter,
        max_correspondences: settings.max_correspondences,
    };
    let correspondences = match_features(source_fpfh, target_fpfh, &match_settings)?;
    registration_ransac_based_on_correspondence(
        source_down,
        target_down,
        &correspondences,
        settings,
    )
}

pub fn registration_ransac_based_on_correspondence(
    source_down: &PointCloud,
    target_down: &PointCloud,
    correspondences: &[Correspondence],
    settings: &GlobalRegistrationSettings,
) -> Result<RegistrationResult, String> {
    if correspondences.len() < settings.ransac_n {
        return Err("not enough correspondences for RANSAC".to_string());
    }

    let data = build_correspondence_matrix(source_down, target_down, correspondences);
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let checkers = normalized_checkers(&settings.checkers, settings.distance_threshold);
    let mut best = RegistrationResult {
        transformation: Matrix4::identity(),
        fitness: 0.0,
        inlier_rmse: f64::MAX,
        correspondence_count: 0,
        iterations: 0,
    };

    let mut max_iterations = settings.max_iterations.max(1);
    let mut iter = 0;
    let total = correspondences.len();

    while iter < max_iterations {
        iter += 1;
        let mut sample = Vec::with_capacity(settings.ransac_n);
        while sample.len() < settings.ransac_n {
            let idx = rng.random_range(0..total);
            if !sample.contains(&idx) {
                sample.push(idx);
            }
        }

        let sample_corr: Vec<(usize, usize)> = sample.iter().map(|&i| correspondences[i]).collect();

        let model = match estimate_rigid_from_sample(&data, &sample) {
            Some(m) => m,
            None => continue,
        };
        let transform = model.to_matrix4();

        if !checkers
            .iter()
            .all(|checker| checker.check_sample(source_down, target_down, &sample_corr, &transform))
        {
            continue;
        }
        let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);

        let normal_cos = normal_cos_threshold(&checkers);
        let mut inliers = Vec::new();
        let mut sum_sq = 0.0;
        for (row_idx, &(si, ti)) in correspondences.iter().enumerate() {
            let p = source_down.point(si);
            let p_trans = r * p + t;
            let q = target_down.point(ti);
            let dist = (p_trans - q).norm();
            if dist <= settings.distance_threshold
                && normal_cos
                    .map(|cos| normal_pair_check(source_down, target_down, (si, ti), cos))
                    .unwrap_or(true)
            {
                inliers.push(row_idx);
                sum_sq += dist * dist;
            }
        }

        if inliers.is_empty() {
            continue;
        }

        let fitness = inliers.len() as f64 / total as f64;
        let rmse = (sum_sq / inliers.len() as f64).sqrt();
        if fitness > best.fitness || (fitness == best.fitness && rmse < best.inlier_rmse) {
            best.transformation = transform;
            best.fitness = fitness;
            best.inlier_rmse = rmse;
            best.correspondence_count = inliers.len();
            best.iterations = iter;

            let inlier_ratio = best.fitness.max(1e-6);
            let denom = 1.0 - inlier_ratio.powi(settings.ransac_n as i32);
            if denom > 0.0 && denom < 1.0 {
                let estimate = (1.0 - settings.confidence).ln() / denom.ln();
                let estimate = estimate.ceil() as usize;
                max_iterations = max_iterations.min(estimate.max(1));
            }
        }
    }

    if best.fitness <= 0.0 {
        return Err("RANSAC failed to find a valid transform".to_string());
    }

    Ok(best)
}

fn weighted_rigid_transform(pairs: &[(Vector3<f64>, Vector3<f64>, f64)]) -> Option<Matrix4<f64>> {
    if pairs.len() < 3 {
        return None;
    }
    let mut sum_w = 0.0;
    let mut c0 = Vector3::zeros();
    let mut c1 = Vector3::zeros();
    for (p, q, w) in pairs {
        if *w <= 0.0 {
            continue;
        }
        sum_w += *w;
        c0 += *w * p;
        c1 += *w * q;
    }
    if sum_w <= 0.0 {
        return None;
    }
    c0 /= sum_w;
    c1 /= sum_w;

    let mut h = Matrix3::zeros();
    for (p, q, w) in pairs {
        if *w <= 0.0 {
            continue;
        }
        let dp = p - c0;
        let dq = q - c1;
        h += *w * (dp * dq.transpose());
    }

    let svd = SVD::new(h, true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let v = vt.transpose();
    let mut r = v * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_fix = v;
        v_fix.column_mut(2).neg_mut();
        r = v_fix * u.transpose();
    }
    let t = c1 - r * c0;

    let mut out = Matrix4::identity();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    out[(0, 3)] = t.x;
    out[(1, 3)] = t.y;
    out[(2, 3)] = t.z;
    Some(out)
}

pub fn registration_fgr_based_on_correspondence(
    source_down: &PointCloud,
    target_down: &PointCloud,
    correspondences: &[Correspondence],
    options: &FastGlobalRegistrationOptions,
) -> Result<RegistrationResult, String> {
    if correspondences.len() < 3 {
        return Err("not enough correspondences for FGR".to_string());
    }

    let mut transform = Matrix4::identity();
    let mut weights = vec![1.0; correspondences.len()];
    let mut mu = options.mu.max(1e-6);

    for _ in 0..options.max_iterations.max(1) {
        let mut pairs = Vec::with_capacity(correspondences.len());
        let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
        for (idx, &(si, ti)) in correspondences.iter().enumerate() {
            let p = source_down.point(si);
            let q = target_down.point(ti);
            let p_trans = r * p + t;
            let dist = (p_trans - q).norm();
            if options.maximum_correspondence_distance > 0.0
                && dist > options.maximum_correspondence_distance
            {
                weights[idx] = 0.0;
                continue;
            }
            let w = weights[idx];
            pairs.push((p, q, w));
        }

        if let Some(next) = weighted_rigid_transform(&pairs) {
            transform = next;
        }

        let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
        for (idx, &(si, ti)) in correspondences.iter().enumerate() {
            let p = source_down.point(si);
            let q = target_down.point(ti);
            let p_trans = r * p + t;
            let dist = (p_trans - q).norm();
            weights[idx] = mu / (mu + dist * dist);
        }

        mu *= 1.0 - options.mu_decay;
        if mu < 1e-6 {
            mu = 1e-6;
        }
    }

    let mut inliers = 0usize;
    let mut sum_sq = 0.0;
    let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
    let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
    for &(si, ti) in correspondences {
        let p = source_down.point(si);
        let q = target_down.point(ti);
        let p_trans = r * p + t;
        let dist = (p_trans - q).norm();
        if options.maximum_correspondence_distance <= 0.0
            || dist <= options.maximum_correspondence_distance
        {
            inliers += 1;
            sum_sq += dist * dist;
        }
    }
    let fitness = inliers as f64 / correspondences.len() as f64;
    let rmse = if inliers == 0 {
        0.0
    } else {
        (sum_sq / inliers as f64).sqrt()
    };

    Ok(RegistrationResult {
        transformation: transform,
        fitness,
        inlier_rmse: rmse,
        correspondence_count: inliers,
        iterations: options.max_iterations,
    })
}

pub fn registration_fgr_based_on_feature_matching(
    source_down: &PointCloud,
    target_down: &PointCloud,
    source_fpfh: &DataMatrix,
    target_fpfh: &DataMatrix,
    options: &FastGlobalRegistrationOptions,
    match_settings: &FeatureMatchSettings,
) -> Result<RegistrationResult, String> {
    let correspondences = match_features(source_fpfh, target_fpfh, match_settings)?;
    registration_fgr_based_on_correspondence(source_down, target_down, &correspondences, options)
}

#[derive(Clone, Debug)]
pub struct RegistrationPipelineResult {
    pub global: RegistrationResult,
    pub local: RegistrationResult,
}

pub struct PointCloudRegistrationPipeline {
    pub preprocess: PreprocessSettings,
    pub global: GlobalRegistrationSettings,
    pub fgr: FastGlobalRegistrationOptions,
    pub global_method: GlobalRegistrationMethod,
    pub local: ColoredIcpPipeline,
}

impl Default for PointCloudRegistrationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl PointCloudRegistrationPipeline {
    pub fn new() -> Self {
        let voxel_size = 0.05;
        let preprocess = PreprocessSettings {
            voxel_size,
            normal_radius: voxel_size * 2.0,
            normal_max_nn: 30,
            fpfh_radius: voxel_size * 5.0,
            fpfh_max_nn: 100,
        };
        let mut global = GlobalRegistrationSettings {
            distance_threshold: voxel_size * 1.5,
            ..GlobalRegistrationSettings::default()
        };
        global.checkers = vec![
            CorrespondenceChecker::EdgeLength { threshold: 0.9 },
            CorrespondenceChecker::Distance {
                threshold: global.distance_threshold,
            },
        ];
        Self {
            preprocess,
            global,
            fgr: FastGlobalRegistrationOptions {
                maximum_correspondence_distance: voxel_size * 0.5,
                ..FastGlobalRegistrationOptions::default()
            },
            global_method: GlobalRegistrationMethod::Ransac,
            local: ColoredIcpPipeline::new(),
        }
    }

    pub fn with_global_method(mut self, method: GlobalRegistrationMethod) -> Self {
        self.global_method = method;
        self
    }

    pub fn run(
        &mut self,
        source: &PointCloud,
        target: &PointCloud,
    ) -> Result<RegistrationPipelineResult, String> {
        let (source_down, source_fpfh) = preprocess_point_cloud(source, &self.preprocess)?;
        let (target_down, target_fpfh) = preprocess_point_cloud(target, &self.preprocess)?;
        let global = match self.global_method {
            GlobalRegistrationMethod::Ransac => registration_ransac_based_on_feature_matching(
                &source_down,
                &target_down,
                &source_fpfh,
                &target_fpfh,
                &self.global,
            )?,
            GlobalRegistrationMethod::FastGlobalRegistration => {
                let match_settings = FeatureMatchSettings {
                    mutual: self.global.mutual_filter,
                    max_correspondences: self.global.max_correspondences,
                };
                registration_fgr_based_on_feature_matching(
                    &source_down,
                    &target_down,
                    &source_fpfh,
                    &target_fpfh,
                    &self.fgr,
                    &match_settings,
                )?
            }
        };

        let local = self.local.run(source, target, global.transformation)?;

        Ok(RegistrationPipelineResult { global, local })
    }
}

fn estimate_median_spacing(cloud: &PointCloud) -> Result<f64, String> {
    let n = cloud.len();
    if n < 2 {
        return Err("point cloud has too few points".to_string());
    }
    let mut index = UsearchNeighborhoodGraph::new(2, 3);
    index.initialize(&cloud.points);
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let neighbors = index.neighbors(i);
        if let Some(&j) = neighbors.first() {
            let d = (cloud.point(i) - cloud.point(j)).norm();
            if d > 0.0 {
                distances.push(d);
            }
        }
    }
    if distances.is_empty() {
        return Err("failed to compute spacing".to_string());
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(distances[distances.len() / 2])
}

pub fn auto_tune_pipeline(
    source: &PointCloud,
    target: &PointCloud,
) -> Result<PointCloudRegistrationPipeline, String> {
    let source_spacing = estimate_median_spacing(source)?;
    let target_spacing = estimate_median_spacing(target)?;
    let base = source_spacing.max(target_spacing);
    if base <= 0.0 {
        return Err("invalid spacing estimate".to_string());
    }

    let voxel_size = base * 2.5;
    let preprocess = PreprocessSettings {
        voxel_size,
        normal_radius: voxel_size * 2.0,
        normal_max_nn: 30,
        fpfh_radius: voxel_size * 5.0,
        fpfh_max_nn: 100,
    };
    let mut global = GlobalRegistrationSettings {
        distance_threshold: voxel_size * 1.5,
        ..GlobalRegistrationSettings::default()
    };
    global.checkers = vec![
        CorrespondenceChecker::EdgeLength { threshold: 0.9 },
        CorrespondenceChecker::Distance {
            threshold: global.distance_threshold,
        },
    ];
    let local = ColoredIcpPipeline::new().with_settings(ColoredIcpSettings {
        distance_threshold: voxel_size * 0.4,
        normal_radius: voxel_size * 2.0,
        gradient_radius: voxel_size * 2.0,
        ..ColoredIcpSettings::default()
    });

    Ok(PointCloudRegistrationPipeline {
        preprocess,
        global,
        fgr: FastGlobalRegistrationOptions {
            maximum_correspondence_distance: voxel_size * 0.5,
            ..FastGlobalRegistrationOptions::default()
        },
        global_method: GlobalRegistrationMethod::Ransac,
        local,
    })
}

pub struct ColoredIcpPipeline {
    downsampler: Box<dyn DownsampleStrategy>,
    normal_estimator: Box<dyn NormalEstimationStrategy>,
    gradient_estimator: Box<dyn ColorGradientEstimationStrategy>,
    correspondence: Box<dyn CorrespondenceStrategy>,
    kernel: Box<dyn IcpKernel>,
    scales: Vec<ColoredIcpScale>,
    criteria: IcpConvergenceCriteria,
    settings: ColoredIcpSettings,
}

impl Default for ColoredIcpPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ColoredIcpPipeline {
    pub fn new() -> Self {
        Self {
            downsampler: Box::new(VoxelDownsample),
            normal_estimator: Box::new(KnnNormalEstimator),
            gradient_estimator: Box::new(KnnColorGradientEstimator),
            correspondence: Box::new(UsearchCorrespondence::new()),
            kernel: Box::new(ColoredPointToPlaneKernel),
            scales: vec![
                ColoredIcpScale {
                    voxel_radius: 0.04,
                    max_iteration: 50,
                },
                ColoredIcpScale {
                    voxel_radius: 0.02,
                    max_iteration: 30,
                },
                ColoredIcpScale {
                    voxel_radius: 0.01,
                    max_iteration: 14,
                },
            ],
            criteria: IcpConvergenceCriteria::default(),
            settings: ColoredIcpSettings::default(),
        }
    }

    pub fn with_scales(mut self, scales: Vec<ColoredIcpScale>) -> Self {
        self.scales = scales;
        self
    }

    pub fn with_criteria(mut self, criteria: IcpConvergenceCriteria) -> Self {
        self.criteria = criteria;
        self
    }

    pub fn with_settings(mut self, settings: ColoredIcpSettings) -> Self {
        self.settings = settings;
        self
    }

    pub fn with_downsampler(mut self, downsampler: Box<dyn DownsampleStrategy>) -> Self {
        self.downsampler = downsampler;
        self
    }

    pub fn with_normal_estimator(mut self, estimator: Box<dyn NormalEstimationStrategy>) -> Self {
        self.normal_estimator = estimator;
        self
    }

    pub fn with_gradient_estimator(
        mut self,
        estimator: Box<dyn ColorGradientEstimationStrategy>,
    ) -> Self {
        self.gradient_estimator = estimator;
        self
    }

    pub fn with_correspondence(mut self, correspondence: Box<dyn CorrespondenceStrategy>) -> Self {
        self.correspondence = correspondence;
        self
    }

    pub fn with_kernel(mut self, kernel: Box<dyn IcpKernel>) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn run(
        &mut self,
        source: &PointCloud,
        target: &PointCloud,
        mut transform: Matrix4<f64>,
    ) -> Result<RegistrationResult, String> {
        let mut total_iterations = 0;
        let mut last_fitness = 0.0;
        let mut last_rmse = 0.0;
        let mut last_count = 0;

        for scale in &self.scales {
            let voxel = scale.voxel_radius;
            let source_down = self.downsampler.downsample(source, voxel)?;
            let mut target_down = self.downsampler.downsample(target, voxel)?;

            target_down = self.normal_estimator.estimate(
                &target_down,
                voxel.max(self.settings.normal_radius) * 2.0,
                self.settings.normal_max_nn,
            )?;
            target_down = self.gradient_estimator.estimate(
                &target_down,
                voxel.max(self.settings.gradient_radius) * 2.0,
                self.settings.gradient_max_nn,
            )?;

            self.correspondence.prepare(&target_down)?;

            let max_iter = scale.max_iteration.min(self.criteria.max_iteration);
            for _ in 0..max_iter {
                let (correspondences, fitness, rmse) =
                    self.find_correspondences(&source_down, &target_down, &transform, voxel)?;

                total_iterations += 1;
                last_count = correspondences.len();
                if correspondences.is_empty() {
                    break;
                }

                if last_fitness > 0.0 {
                    let rel_fit = (fitness - last_fitness).abs() / last_fitness.max(1e-12);
                    let rel_rmse = (rmse - last_rmse).abs() / last_rmse.max(1e-12);
                    if rel_fit < self.criteria.relative_fitness
                        && rel_rmse < self.criteria.relative_rmse
                    {
                        last_fitness = fitness;
                        last_rmse = rmse;
                        break;
                    }
                }
                last_fitness = fitness;
                last_rmse = rmse;

                let delta = self.kernel.solve(
                    &source_down,
                    &target_down,
                    &correspondences,
                    &transform,
                    &self.settings,
                );
                let delta = match delta {
                    Some(d) => d,
                    None => break,
                };
                transform = delta * transform;
            }
        }

        Ok(RegistrationResult {
            transformation: transform,
            fitness: last_fitness,
            inlier_rmse: last_rmse,
            correspondence_count: last_count,
            iterations: total_iterations,
        })
    }

    fn find_correspondences(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        transform: &Matrix4<f64>,
        voxel: f64,
    ) -> Result<(Vec<Correspondence>, f64, f64), String> {
        let r = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
        let threshold = if self.settings.distance_threshold > 0.0 {
            self.settings.distance_threshold
        } else {
            voxel
        };
        let threshold_sq = threshold * threshold;

        let normals = match &target.normals {
            Some(n) => n,
            None => return Err("target normals are missing".to_string()),
        };

        let mut correspondences = Vec::new();
        let mut sum_sq = 0.0;
        for i in 0..source.len() {
            let p = source.point(i);
            let p_trans = r * p + t;
            if let Some((j, dist_sq)) = self.correspondence.nearest(&p_trans)
                && dist_sq <= threshold_sq
            {
                correspondences.push((i, j));
                let q = Vector3::new(
                    target.points[(j, 0)],
                    target.points[(j, 1)],
                    target.points[(j, 2)],
                );
                let n = Vector3::new(normals[(j, 0)], normals[(j, 1)], normals[(j, 2)]);
                let r_g = n.dot(&(p_trans - q));
                sum_sq += r_g * r_g;
            }
        }
        let fitness = if !source.is_empty() {
            correspondences.len() as f64 / source.len() as f64
        } else {
            0.0
        };
        let rmse = if correspondences.is_empty() {
            0.0
        } else {
            (sum_sq / correspondences.len() as f64).sqrt()
        };

        Ok((correspondences, fitness, rmse))
    }
}
