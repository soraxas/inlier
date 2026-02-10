//! Core shared types for the SupeRANSAC Rust port.
//!
//! The goal here is to mirror the conceptual role of the C++ `DataMatrix`
//! and related aliases, without prematurely committing to a specific memory
//! layout or set of helpers. We start with simple `nalgebra` aliases that can
//! be extended or optimized later.

use nalgebra::{DMatrix as NaDMatrix, DVector, Vector2, Vector3};
use std::ops::{Deref, DerefMut};

/// Opaque wrapper around a dynamic `f64` matrix representing data points.
///
/// Internally stores data where each row is a point. All access should go through
/// wrapper methods to allow future layout changes without breaking API.
#[derive(Clone, Debug)]
pub struct DataMatrix(NaDMatrix<f64>);

impl DataMatrix {
    /// Create a zero-initialized matrix with the given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self(NaDMatrix::<f64>::zeros(rows, cols))
    }

    /// Number of data points (rows).
    pub fn n_points(&self) -> usize {
        self.0.nrows()
    }

    /// Number of dimensions per point (columns).
    pub fn n_dims(&self) -> usize {
        self.0.ncols()
    }

    /// Create from row-major slice: [point0_data..., point1_data..., ...].
    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        Self(NaDMatrix::from_row_slice(rows, cols, data))
    }

    /// Get a single element at (point_idx, dim_idx).
    #[inline]
    pub fn get(&self, point_idx: usize, dim_idx: usize) -> f64 {
        self.0[(point_idx, dim_idx)]
    }

    /// Set a single element at (point_idx, dim_idx).
    #[inline]
    pub fn set(&mut self, point_idx: usize, dim_idx: usize, value: f64) {
        self.0[(point_idx, dim_idx)] = value;
    }

    /// Get a point as a dynamically-sized vector.
    pub fn get_point(&self, point_idx: usize) -> DVector<f64> {
        self.0.row(point_idx).transpose()
    }

    /// Get a specific dimension (column) for a point.
    #[inline]
    pub fn get_dim(&self, point_idx: usize, dim_idx: usize) -> f64 {
        self.get(point_idx, dim_idx)
    }

    /// Set a specific dimension (column) for a point.
    #[inline]
    pub fn set_dim(&mut self, point_idx: usize, dim_idx: usize, value: f64) {
        self.set(point_idx, dim_idx, value)
    }

    /// Extract subslice of dimensions for a point (useful for source/target splits).
    /// Returns a vector from [dim_start, dim_end).
    pub fn get_point_slice(
        &self,
        point_idx: usize,
        dim_start: usize,
        dim_end: usize,
    ) -> DVector<f64> {
        let n = dim_end - dim_start;
        DVector::from_fn(n, |i, _| self.get(point_idx, dim_start + i))
    }

    /// Iterate over all points as dynamic vectors.
    pub fn iter_points(&self) -> impl Iterator<Item = DVector<f64>> + '_ {
        (0..self.n_points()).map(move |i| self.get_point(i))
    }

    /// Filter rows based on a boolean mask.
    pub fn filter_points(&self, mask: &[bool]) -> Self {
        assert_eq!(
            mask.len(),
            self.n_points(),
            "Mask length must match number of points"
        );

        let cols = self.n_dims();
        let mut flat = Vec::new();

        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                for c in 0..cols {
                    flat.push(self.get(i, c));
                }
            }
        }

        let rows = flat.len() / cols;
        Self::from_row_slice(rows, cols, &flat)
    }

    /// Convert to owned nalgebra matrix (for interop with existing code).
    pub fn into_inner(self) -> NaDMatrix<f64> {
        self.0
    }

    /// Borrow the inner nalgebra matrix (try to avoid this in new code).
    pub fn as_inner(&self) -> &NaDMatrix<f64> {
        &self.0
    }

    /// Mutable borrow of inner matrix (try to avoid this in new code).
    pub fn as_inner_mut(&mut self) -> &mut NaDMatrix<f64> {
        &mut self.0
    }

    #[deprecated(since = "0.1.0", note = "Use direct methods on DataMatrix instead")]
    /// Borrow this matrix as a point view.
    pub fn as_points(&self) -> DataPoints<'_> {
        DataPoints { mat: &self.0 }
    }
}

// Keep Deref/DerefMut temporarily for gradual migration
// TODO: Remove these once all direct access is replaced
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
/// DEPRECATED: This wrapper is being phased out. Use DataMatrix methods directly.
#[deprecated(since = "0.1.0", note = "Use DataMatrix methods directly")]
pub struct DataPoints<'a> {
    mat: &'a NaDMatrix<f64>,
}

#[allow(deprecated)]
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
