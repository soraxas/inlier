//! Core shared types for the SupeRANSAC Rust port.
//!
//! The goal here is to mirror the conceptual role of the C++ `DataMatrix`
//! and related aliases, without prematurely committing to a specific memory
//! layout or set of helpers. We start with simple `nalgebra` aliases that can
//! be extended or optimized later.

use nalgebra::{DMatrix as NaDMatrix, Vector2, Vector3};
use std::ops::{Deref, DerefMut};

/// Opaque wrapper around a dynamic `f64` matrix representing data points.
#[derive(Clone, Debug)]
pub struct DataMatrix(NaDMatrix<f64>);

impl DataMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self(NaDMatrix::<f64>::zeros(rows, cols))
    }

    pub fn n_points(&self) -> usize {
        self.0.nrows()
    }

    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        Self(NaDMatrix::from_row_slice(rows, cols, data))
    }

    pub fn into_inner(self) -> NaDMatrix<f64> {
        self.0
    }

    // pub fn iter_points(&self) -> nalgebra::iter::ColumnIter<'_, f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> {
    //     self.0.column_iter()
    // }

    #[deprecated(since = "0.1.0", note = "Use as_points_view instead")]
    /// Borrow this matrix as a point view.
    pub fn as_points(&self) -> DataPoints<'_> {
        DataPoints { mat: &self.0 }
    }
}

impl Deref for DataMatrix {
    type Target = NaDMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DataMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<NaDMatrix<f64>> for DataMatrix {
    fn from(m: NaDMatrix<f64>) -> Self {
        Self(m)
    }
}

impl From<DataMatrix> for NaDMatrix<f64> {
    fn from(m: DataMatrix) -> Self {
        m.0
    }
}

/// Borrowed view over point correspondences.
///
/// This wrapper keeps the underlying storage opaque so we can change the
/// representation later while exposing point-wise iterators today.
pub struct DataPoints<'a> {
    mat: &'a NaDMatrix<f64>,
}

impl<'a> DataPoints<'a> {
    /// Number of points (rows in the current layout).
    pub fn len(&self) -> usize {
        self.mat.nrows()
    }

    /// Get a single point as an array `[sx, sy, sz, tx, ty, tz]` if available.
    pub fn point(&self, idx: usize) -> Option<[f64; 6]> {
        if idx >= self.mat.nrows() || self.mat.ncols() < 6 {
            return None;
        }
        Some([
            self.mat[(idx, 0)],
            self.mat[(idx, 1)],
            self.mat[(idx, 2)],
            self.mat[(idx, 3)],
            self.mat[(idx, 4)],
            self.mat[(idx, 5)],
        ])
    }

    /// Iterate over points as small arrays.
    pub fn iter_points(&self) -> impl Iterator<Item = [f64; 6]> + '_ {
        (0..self.mat.nrows()).filter_map(|i| self.point(i))
    }

    /// Access the raw matrix if needed for legacy code.
    pub fn as_matrix(&self) -> &'a NaDMatrix<f64> {
        self.mat
    }
}

/// 2D point used in many geometric problems.
pub type Point2 = Vector2<f64>;

/// 3D point / vector used in pose and rigid transform problems.
pub type Point3 = Vector3<f64>;
