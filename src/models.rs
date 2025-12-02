//! Geometric models used by SupeRANSAC.
//!
//! This module defines lightweight model types mirroring the C++ `models::Model`
//! hierarchy: homographies, fundamental/essential matrices, absolute pose, and
//! rigid transformations.

use nalgebra::{Matrix3, Matrix4, Rotation3, Translation3, UnitQuaternion, Vector3};

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
        Self { rotation, translation }
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
        Self { rotation, translation }
    }

    pub fn to_matrix4(&self) -> Matrix4<f64> {
        let r = self.rotation.to_homogeneous();
        let t = self.translation.to_homogeneous();
        r * t
    }
}
