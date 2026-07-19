//! Geometric models used by SupeRANSAC.
//!
//! This module defines lightweight model types mirroring the C++ `models::Model`
//! hierarchy: homographies, fundamental/essential matrices, absolute pose, and
//! rigid transformations.

use nalgebra::{Matrix3, Matrix4, Rotation3, Translation3, UnitQuaternion, Vector3, Vector4};

/// Planar projective transformation represented by a 3x3 matrix.
#[derive(Clone, Debug)]
pub struct Homography {
    pub h: Matrix3<f64>,
}

impl Homography {
    pub fn new(h: Matrix3<f64>) -> Self {
        Self { h }
    }
}

/// Fundamental matrix relating two pinhole views.
#[derive(Clone, Debug)]
pub struct FundamentalMatrix {
    pub f: Matrix3<f64>,
}

impl FundamentalMatrix {
    pub fn new(f: Matrix3<f64>) -> Self {
        Self { f }
    }
}

/// Essential matrix relating two calibrated views.
#[derive(Clone, Debug)]
pub struct EssentialMatrix {
    pub e: Matrix3<f64>,
}

impl EssentialMatrix {
    pub fn new(e: Matrix3<f64>) -> Self {
        Self { e }
    }
}

/// Absolute camera pose: rotation and translation from world to camera.
#[derive(Clone, Debug)]
pub struct AbsolutePose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Translation3<f64>,
}

impl AbsolutePose {
    pub fn new(rotation: UnitQuaternion<f64>, translation: Translation3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn from_rt(r: Matrix3<f64>, t: Vector3<f64>) -> Self {
        let rot = Rotation3::from_matrix_unchecked(r);
        let quat = UnitQuaternion::from_rotation_matrix(&rot);
        let tr = Translation3::from(t);
        Self::new(quat, tr)
    }
}

/// Rigid transform in 3D (rotation + translation).
#[derive(Clone, Debug)]
pub struct RigidTransform {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Translation3<f64>,
}

impl RigidTransform {
    pub fn new(rotation: UnitQuaternion<f64>, translation: Translation3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn to_matrix4(&self) -> Matrix4<f64> {
        let r = self.rotation.to_homogeneous();
        let t = self.translation.to_homogeneous();
        t * r
    }
}

/// Similarity transform in 3D (scale + rotation + translation).
#[derive(Clone, Debug)]
pub struct SimilarityTransform {
    pub scale: f64,
    pub rotation: UnitQuaternion<f64>,
    pub translation: Translation3<f64>,
}

impl SimilarityTransform {
    pub fn new(scale: f64, rotation: UnitQuaternion<f64>, translation: Translation3<f64>) -> Self {
        Self {
            scale,
            rotation,
            translation,
        }
    }

    pub fn to_matrix4(&self) -> Matrix4<f64> {
        let r = self.rotation.to_homogeneous();
        let t = self.translation.to_homogeneous();
        // Apply uniform scale.
        let mut m = r;
        m.fixed_view_mut::<3, 3>(0, 0)
            .apply(|val| *val *= self.scale);
        t * m
    }
}

/// Line in 2D represented as ax + by + c = 0, normalized so that a² + b² = 1.
///
/// The line parameters are stored as [a, b, c] where:
/// - (a, b) is the unit normal vector
/// - c is the distance from origin (with sign)
#[derive(Clone, Debug)]
pub struct Line {
    /// Line parameters [a, b, c] where ax + by + c = 0
    pub params: Vector3<f64>,
}

impl Line {
    /// Create a new line from parameters [a, b, c].
    /// The parameters will be normalized so that a² + b² = 1.
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        let norm = (a * a + b * b).sqrt();
        if norm < 1e-10 {
            // Degenerate line (both a and b are zero)
            Self {
                params: Vector3::new(0.0, 1.0, c),
            }
        } else {
            Self {
                params: Vector3::new(a / norm, b / norm, c / norm),
            }
        }
    }

    /// Create a line from parameters, assuming they're already normalized.
    pub fn from_normalized(params: Vector3<f64>) -> Self {
        Self { params }
    }

    /// Get the line parameters as [a, b, c].
    pub fn params(&self) -> &Vector3<f64> {
        &self.params
    }

    /// Compute the distance from a point (x, y) to this line.
    pub fn distance_to_point(&self, x: f64, y: f64) -> f64 {
        (self.params[0] * x + self.params[1] * y + self.params[2]).abs()
    }

    /// Convert to slope-intercept form (y = mx + b), if possible.
    /// Returns None if the line is vertical (b ≈ 0).
    pub fn to_slope_intercept(&self) -> Option<(f64, f64)> {
        if self.params[1].abs() < 1e-10 {
            None // Vertical line
        } else {
            let slope = -self.params[0] / self.params[1];
            let intercept = -self.params[2] / self.params[1];
            Some((slope, intercept))
        }
    }
}

/// 3-D plane model: `ax + by + cz + d = 0` with `|normal| = 1`.
#[derive(Clone, Debug)]
pub struct Plane3 {
    /// Unit normal vector (a, b, c).
    pub normal: Vector3<f64>,
    /// Plane offset d, so that `normal · p + d = 0` for any point p on the plane.
    pub d: f64,
}

impl Plane3 {
    /// Construct from coefficients, normalising the normal on the fly.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        let norm = (a * a + b * b + c * c).sqrt();
        if norm < 1e-10 {
            Self {
                normal: Vector3::new(0.0, 0.0, 1.0),
                d: 0.0,
            }
        } else {
            Self {
                normal: Vector3::new(a / norm, b / norm, c / norm),
                d: d / norm,
            }
        }
    }

    /// Construct from a unit normal and a point known to lie on the plane.
    pub fn from_normal_and_point(normal: Vector3<f64>, point: Vector3<f64>) -> Self {
        let d = -normal.dot(&point);
        Self { normal, d }
    }

    /// Signed point-to-plane distance (positive on the normal side).
    #[inline]
    pub fn signed_distance(&self, x: f64, y: f64, z: f64) -> f64 {
        self.normal.x * x + self.normal.y * y + self.normal.z * z + self.d
    }

    /// Unsigned point-to-plane distance.
    #[inline]
    pub fn distance(&self, x: f64, y: f64, z: f64) -> f64 {
        self.signed_distance(x, y, z).abs()
    }

    /// Plane parameters as a 4-vector [a, b, c, d].
    pub fn to_vec4(&self) -> Vector4<f64> {
        Vector4::new(self.normal.x, self.normal.y, self.normal.z, self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_matrix_round_trip_preserves_points() {
        let transform = RigidTransform::new(
            UnitQuaternion::from_euler_angles(0.2, -0.3, 0.4),
            Translation3::new(2.0, -1.0, 0.5),
        );
        let point = Vector4::new(0.7, -1.2, 3.0, 1.0);
        let matrix = transform.to_matrix4();
        let recovered =
            matrix.try_inverse().expect("rigid transform is invertible") * (matrix * point);

        assert!((recovered - point).norm() < 1e-12);
        assert!(
            (matrix.fixed_view::<3, 1>(0, 3).into_owned() - transform.translation.vector).norm()
                < 1e-12
        );
    }

    #[test]
    fn similarity_matrix_round_trip_preserves_points() {
        let transform = SimilarityTransform::new(
            1.75,
            UnitQuaternion::from_euler_angles(-0.1, 0.25, 0.3),
            Translation3::new(-0.5, 1.5, 2.0),
        );
        let point = Vector4::new(1.0, 2.0, -0.5, 1.0);
        let matrix = transform.to_matrix4();
        let recovered =
            matrix.try_inverse().expect("non-zero scale is invertible") * (matrix * point);

        assert!((recovered - point).norm() < 1e-12);
        assert!(
            (matrix.fixed_view::<3, 1>(0, 3).into_owned() - transform.translation.vector).norm()
                < 1e-12
        );
    }
}
