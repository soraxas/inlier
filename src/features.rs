//! Feature extraction and matching for point cloud registration.

use nalgebra::Vector3;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::pointcloud::PointCloud;
use crate::types::DataMatrix;

#[derive(Clone, Debug)]
pub struct FpfhSettings {
    pub radius: f64,
    pub max_nn: usize,
}

impl Default for FpfhSettings {
    fn default() -> Self {
        Self {
            radius: 0.25,
            max_nn: 100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeatureMatchSettings {
    pub mutual: bool,
    pub max_correspondences: usize,
}

impl Default for FeatureMatchSettings {
    fn default() -> Self {
        Self {
            mutual: true,
            max_correspondences: 0,
        }
    }
}

struct UsearchKnnIndex {
    index: Index,
    dims: usize,
}

impl UsearchKnnIndex {
    fn new(dims: usize, reserve: usize) -> Result<Self, String> {
        let options = IndexOptions {
            dimensions: dims,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options).map_err(|_| "failed to create index".to_string())?;
        index
            .reserve(reserve)
            .map_err(|_| "failed to reserve index".to_string())?;
        Ok(Self { index, dims })
    }

    fn add_point(&self, key: usize, vec: &[f64]) -> Result<(), String> {
        if vec.len() != self.dims {
            return Err("vector dimension mismatch".to_string());
        }
        let mut data = Vec::with_capacity(self.dims);
        for &v in vec {
            data.push(v as f32);
        }
        self.index
            .add(key as u64, &data)
            .map_err(|_| "failed to add vector".to_string())
    }

    fn search(&self, vec: &[f64], k: usize) -> Result<Vec<(usize, f64)>, String> {
        if vec.len() != self.dims {
            return Err("vector dimension mismatch".to_string());
        }
        let mut data = Vec::with_capacity(self.dims);
        for &v in vec {
            data.push(v as f32);
        }
        let results = self
            .index
            .search(&data, k)
            .map_err(|_| "search failed".to_string())?;
        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&key, &dist)| (key as usize, dist as f64))
            .collect())
    }
}

fn histogram_bin(value: f64, min: f64, max: f64, bins: usize) -> usize {
    if bins == 0 {
        return 0;
    }
    let mut v = value;
    if v < min {
        v = min;
    }
    if v > max {
        v = max;
    }
    let span = max - min;
    if span <= 0.0 {
        return 0;
    }
    let scaled = (v - min) / span;
    let idx = (scaled * bins as f64) as usize;
    idx.min(bins - 1)
}

fn normalize_hist(hist: &mut [f64]) {
    let sum: f64 = hist.iter().sum();
    if sum > 0.0 {
        for h in hist.iter_mut() {
            *h /= sum;
        }
    }
}

fn build_point_index(points: &DataMatrix) -> Result<UsearchKnnIndex, String> {
    let index = UsearchKnnIndex::new(3, points.nrows())?;
    for i in 0..points.nrows() {
        let vec = [points[(i, 0)], points[(i, 1)], points[(i, 2)]];
        index.add_point(i, &vec)?;
    }
    Ok(index)
}

fn neighbors_within(
    index: &UsearchKnnIndex,
    points: &DataMatrix,
    center: usize,
    radius: f64,
    max_nn: usize,
) -> Vec<usize> {
    let query = [
        points[(center, 0)],
        points[(center, 1)],
        points[(center, 2)],
    ];
    let k = (max_nn + 1).max(2).min(points.nrows());
    let radius_sq = radius * radius;
    let mut neighbors = Vec::new();
    if let Ok(results) = index.search(&query, k) {
        for (idx, dist) in results {
            if idx == center {
                continue;
            }
            if radius <= 0.0 || dist <= radius_sq {
                neighbors.push(idx);
            }
        }
    }
    neighbors
}

pub fn compute_fpfh_features(
    cloud: &PointCloud,
    settings: &FpfhSettings,
) -> Result<DataMatrix, String> {
    let normals = cloud
        .normals
        .as_ref()
        .ok_or_else(|| "normals are required for FPFH".to_string())?;
    if cloud.is_empty() {
        return Ok(DataMatrix::zeros(0, 33));
    }

    let index = build_point_index(&cloud.points)?;
    let bins = 11;
    let mut spfh = vec![vec![0.0_f64; 33]; cloud.len()];

    for i in 0..cloud.len() {
        let neighbors =
            neighbors_within(&index, &cloud.points, i, settings.radius, settings.max_nn);
        if neighbors.is_empty() {
            continue;
        }
        let p = Vector3::new(
            cloud.points[(i, 0)],
            cloud.points[(i, 1)],
            cloud.points[(i, 2)],
        );
        let n_i = Vector3::new(normals[(i, 0)], normals[(i, 1)], normals[(i, 2)]);
        if n_i.norm() < 1e-9 {
            continue;
        }

        let mut hist_alpha = vec![0.0_f64; bins];
        let mut hist_phi = vec![0.0_f64; bins];
        let mut hist_theta = vec![0.0_f64; bins];

        for &j in &neighbors {
            let q = Vector3::new(
                cloud.points[(j, 0)],
                cloud.points[(j, 1)],
                cloud.points[(j, 2)],
            );
            let n_j = Vector3::new(normals[(j, 0)], normals[(j, 1)], normals[(j, 2)]);
            let d = q - p;
            let d_norm = d.norm();
            if d_norm < 1e-12 {
                continue;
            }
            let u = n_i;
            let v = u.cross(&d);
            if v.norm() < 1e-12 {
                continue;
            }
            let v = v.normalize();
            let w = u.cross(&v);

            let alpha = v.dot(&n_j);
            let phi = u.dot(&d) / d_norm;
            let theta = w.dot(&n_j).atan2(u.dot(&n_j));

            let ia = histogram_bin(alpha, -1.0, 1.0, bins);
            let ip = histogram_bin(phi, -1.0, 1.0, bins);
            let it = histogram_bin(theta, -std::f64::consts::PI, std::f64::consts::PI, bins);
            hist_alpha[ia] += 1.0;
            hist_phi[ip] += 1.0;
            hist_theta[it] += 1.0;
        }

        normalize_hist(&mut hist_alpha);
        normalize_hist(&mut hist_phi);
        normalize_hist(&mut hist_theta);

        let mut out = vec![0.0_f64; 33];
        out[..bins].copy_from_slice(&hist_alpha);
        out[bins..2 * bins].copy_from_slice(&hist_phi);
        out[2 * bins..3 * bins].copy_from_slice(&hist_theta);
        spfh[i] = out;
    }

    let mut fpfh = DataMatrix::zeros(cloud.len(), 33);
    for i in 0..cloud.len() {
        let neighbors =
            neighbors_within(&index, &cloud.points, i, settings.radius, settings.max_nn);
        if neighbors.is_empty() {
            for k in 0..33 {
                fpfh[(i, k)] = spfh[i][k];
            }
            continue;
        }

        let p = Vector3::new(
            cloud.points[(i, 0)],
            cloud.points[(i, 1)],
            cloud.points[(i, 2)],
        );
        let mut accum = vec![0.0_f64; 33];
        let mut weight_sum = 0.0_f64;

        for &j in &neighbors {
            let q = Vector3::new(
                cloud.points[(j, 0)],
                cloud.points[(j, 1)],
                cloud.points[(j, 2)],
            );
            let dist = (q - p).norm();
            if dist < 1e-12 {
                continue;
            }
            let w = 1.0 / dist;
            weight_sum += w;
            for k in 0..33 {
                accum[k] += w * spfh[j][k];
            }
        }

        for k in 0..33 {
            let mut value = spfh[i][k];
            if weight_sum > 0.0 {
                value += accum[k] / weight_sum;
            }
            fpfh[(i, k)] = value;
        }
    }

    Ok(fpfh)
}

pub fn match_features(
    source: &DataMatrix,
    target: &DataMatrix,
    settings: &FeatureMatchSettings,
) -> Result<Vec<(usize, usize)>, String> {
    if source.ncols() != 33 || target.ncols() != 33 {
        return Err("FPFH feature matrices must be Nx33".to_string());
    }
    let index_t = UsearchKnnIndex::new(33, target.nrows())?;
    for i in 0..target.nrows() {
        let mut vec = [0.0_f64; 33];
        for k in 0..33 {
            vec[k] = target[(i, k)];
        }
        index_t.add_point(i, &vec)?;
    }

    let mut reverse = Vec::new();
    if settings.mutual {
        let index_s = UsearchKnnIndex::new(33, source.nrows())?;
        for i in 0..source.nrows() {
            let mut vec = [0.0_f64; 33];
            for k in 0..33 {
                vec[k] = source[(i, k)];
            }
            index_s.add_point(i, &vec)?;
        }
        reverse.resize(target.nrows(), usize::MAX);
        for j in 0..target.nrows() {
            let mut vec = [0.0_f64; 33];
            for k in 0..33 {
                vec[k] = target[(j, k)];
            }
            if let Ok(results) = index_s.search(&vec, 1)
                && let Some((idx, _)) = results.first()
            {
                reverse[j] = *idx;
            }
        }
    }

    let mut matches = Vec::new();
    for i in 0..source.nrows() {
        let mut vec = [0.0_f64; 33];
        for k in 0..33 {
            vec[k] = source[(i, k)];
        }
        let results = index_t.search(&vec, 1)?;
        if let Some((j, _)) = results.first() {
            if settings.mutual && (*j >= reverse.len() || reverse[*j] != i) {
                continue;
            }
            matches.push((i, *j));
            if settings.max_correspondences > 0 && matches.len() >= settings.max_correspondences {
                break;
            }
        }
    }

    Ok(matches)
}
