//! Geometric models used by SupeRANSAC.
//!
//! For now we define only a homography model; additional models
//! (fundamental, essential, poses, rigid transforms) will be added later.

use nalgebra::Matrix3;

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
