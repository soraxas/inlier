//! Point cloud registration utilities and colored ICP pipeline.
use std::collections::HashMap;

use nalgebra::{Matrix3, Matrix4, SMatrix, SVector, SymmetricEigen, Vector3};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::types::DataMatrix;

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
        if let Some(colors) = &colors {
            if colors.ncols() != 3 || colors.nrows() != points.nrows() {
                return Err("colors must be Nx3 matrix aligned to points".to_string());
            }
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
            if use_colors {
                if let Some(c) = self.color(i) {
                    entry.color_sum += c;
                }
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
        let mut query = [point.x as f32, point.y as f32, point.z as f32];
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

pub trait IcpKernel {
    fn solve(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        correspondences: &[(usize, usize)],
        transform: &Matrix4<f64>,
        settings: &ColoredIcpSettings,
    ) -> Option<Matrix4<f64>>;
}

pub struct ColoredPointToPlaneKernel;

impl ColoredPointToPlaneKernel {
    fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    fn se3_exp(delta: &SVector<f64, 6>) -> Matrix4<f64> {
        let w = Vector3::new(delta[0], delta[1], delta[2]);
        let v = Vector3::new(delta[3], delta[4], delta[5]);
        let theta = w.norm();
        let mut r = Matrix3::identity();
        let mut v_mat = Matrix3::identity();
        if theta > 1e-12 {
            let w_hat = Self::skew(&w);
            let w_hat2 = w_hat * w_hat;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            r = Matrix3::identity()
                + (sin_t / theta) * w_hat
                + ((1.0 - cos_t) / (theta * theta)) * w_hat2;
            v_mat = Matrix3::identity()
                + ((1.0 - cos_t) / (theta * theta)) * w_hat
                + ((theta - sin_t) / (theta * theta * theta)) * w_hat2;
        } else {
            let w_hat = Self::skew(&w);
            r = Matrix3::identity() + w_hat;
            v_mat = Matrix3::identity() + 0.5 * w_hat;
        }
        let t = v_mat * v;
        let mut out = Matrix4::identity();
        out.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        out[(0, 3)] = t.x;
        out[(1, 3)] = t.y;
        out[(2, 3)] = t.z;
        out
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
            let w = settings.lambda_geometric;
            ata += w * (j * j.transpose());
            atb += w * (j * (-r_g));

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
                    let wc = settings.lambda_color;
                    ata += wc * (j_c * j_c.transpose());
                    atb += wc * (j_c * (-r_c));
                }
            }
        }

        let delta = ata.lu().solve(&atb)?;
        Some(Self::se3_exp(&delta))
    }
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
            let mut source_down = self.downsampler.downsample(source, voxel)?;
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
                let (correspondences, fitness, rmse) = match self.find_correspondences(
                    &source_down,
                    &target_down,
                    &transform,
                    voxel,
                ) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };

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
    ) -> Result<(Vec<(usize, usize)>, f64, f64), String> {
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
            if let Some((j, dist_sq)) = self.correspondence.nearest(&p_trans) {
                if dist_sq <= threshold_sq {
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
        }
        let fitness = if source.len() > 0 {
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
