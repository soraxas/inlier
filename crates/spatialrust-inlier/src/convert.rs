use inlier::types::DataMatrix;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use spatialrust_core::{
    HasNormals3, HasPositions3, PointCloud, PointCloudBuilder, SpatialError, SpatialResult,
};
use spatialrust_math::{Isometry3, Quat, Vec3};

/// Converts a `PointCloud` to an inlier `DataMatrix` (xyz columns, upcast to f64).
///
/// Returns an error if the cloud has no position fields.
pub fn point_cloud_to_data_matrix(cloud: &PointCloud) -> SpatialResult<DataMatrix> {
    let (xs, ys, zs) = cloud.positions3()?;
    let n = xs.len();
    // DataMatrix::from_row_slice expects [p0x, p0y, p0z, p1x, p1y, p1z, ...]
    let mut data = Vec::with_capacity(3 * n);
    for i in 0..n {
        data.push(xs[i] as f64);
        data.push(ys[i] as f64);
        data.push(zs[i] as f64);
    }
    Ok(DataMatrix::from_row_slice(n, 3, &data))
}

/// Converts a 3-dimensional inlier `DataMatrix` back to an xyz `PointCloud` (downcast to f32).
///
/// Panics if `m.n_dims() != 3`.
pub fn data_matrix_to_point_cloud(m: &DataMatrix) -> SpatialResult<PointCloud> {
    assert_eq!(m.n_dims(), 3, "DataMatrix must be 3-dimensional for xyz conversion");
    let mut builder = PointCloudBuilder::xyz();
    for pt in m.iter_points() {
        builder
            .push_point([pt[0] as f32, pt[1] as f32, pt[2] as f32])
            .map_err(|e| SpatialError::InvalidArgument(e.to_string()))?;
    }
    builder.build()
}

/// Converts a cloud with normals to a 6-column `DataMatrix` `[x,y,z,nx,ny,nz]` (f64).
///
/// Use this when you want inlier's NAPSAC or importance samplers to exploit
/// normal information for spatially coherent sampling.
///
/// Returns an error if the cloud lacks position or normal fields.
pub fn point_cloud_with_normals_to_data_matrix(cloud: &PointCloud) -> SpatialResult<DataMatrix> {
    let (xs, ys, zs) = cloud.positions3()?;
    let (nxs, nys, nzs) = cloud.normals3()?;
    let n = xs.len();
    let mut data = Vec::with_capacity(6 * n);
    for i in 0..n {
        data.push(xs[i] as f64);
        data.push(ys[i] as f64);
        data.push(zs[i] as f64);
        data.push(nxs[i] as f64);
        data.push(nys[i] as f64);
        data.push(nzs[i] as f64);
    }
    Ok(DataMatrix::from_row_slice(n, 6, &data))
}

/// Converts a nalgebra rotation matrix + translation vector (f64) to a
/// SpatialRust `Isometry3<f32>`.
pub fn nalgebra_to_isometry3(
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
) -> Isometry3<f32> {
    let rot3 = Rotation3::from_matrix_unchecked(*rotation);
    let uq: UnitQuaternion<f64> = UnitQuaternion::from_rotation_matrix(&rot3);
    let q = Quat { x: uq.i as f32, y: uq.j as f32, z: uq.k as f32, w: uq.w as f32 };
    let t = Vec3::new(translation.x as f32, translation.y as f32, translation.z as f32);
    Isometry3::new(q, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::StandardSchemas;

    #[test]
    fn round_trip_xyz() {
        let mut builder = spatialrust_core::PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([1.0f32, 2.0, 3.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0]).unwrap();
        let cloud = builder.build().unwrap();

        let dm = point_cloud_to_data_matrix(&cloud).unwrap();
        assert_eq!(dm.n_points(), 2);
        assert_eq!(dm.n_dims(), 3);
        assert!((dm.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((dm.get(1, 2) - 6.0).abs() < 1e-6);

        let cloud2 = data_matrix_to_point_cloud(&dm).unwrap();
        assert_eq!(cloud2.len(), 2);
    }

    #[test]
    fn isometry3_identity() {
        let rot = Matrix3::identity();
        let trans = Vector3::zeros();
        let iso = nalgebra_to_isometry3(&rot, &trans);
        assert!((iso.rotation().w - 1.0).abs() < 1e-6);
        assert!(iso.translation().x.abs() < 1e-6);
    }
}
