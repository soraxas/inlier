//! Core shared types for the SupeRANSAC Rust port.
//!
//! The goal here is to mirror the conceptual role of the C++ `DataMatrix`
//! and related aliases, without prematurely committing to a specific memory
//! layout or set of helpers. We start with simple `nalgebra` aliases that can
//! be extended or optimized later.

use nalgebra::{DMatrix as NaDMatrix, DVector, Vector2, Vector3};

/// Opaque wrapper around a dynamic `f64` matrix representing data points.
///
/// Internally stores data in COLUMN-MAJOR layout where each column is a point.
/// This provides better cache locality when iterating over points with nalgebra.
/// All access goes through wrapper methods to keep this internal detail hidden.
#[derive(Clone, Debug)]
pub struct DataMatrix(NaDMatrix<f64>);

impl DataMatrix {
    /// Create a zero-initialized matrix with the given dimensions.
    /// Note: internally stored as (dims × points) for column-major layout.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self(NaDMatrix::<f64>::zeros(cols, rows))
    }

    /// Number of data points (columns in internal storage).
    pub fn n_points(&self) -> usize {
        self.0.ncols()
    }

    /// Number of dimensions per point (rows in internal storage).
    pub fn n_dims(&self) -> usize {
        self.0.nrows()
    }

    /// Create from row-major slice: [point0_data..., point1_data..., ...].
    /// Zero-copy: directly interprets as column-major transposed layout.
    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        // OPTIMIZATION: Instead of creating N×M matrix then transposing (copy),
        // we directly create M×N from column-slice (zero-copy reinterpretation).
        // NumPy N×M row-major has same memory as M×N column-major!
        Self(NaDMatrix::from_column_slice(cols, rows, data))
    }

    /// Get a single element at (point_idx, dim_idx).
    #[inline]
    pub fn get(&self, point_idx: usize, dim_idx: usize) -> f64 {
        self.0[(dim_idx, point_idx)]
    }

    /// Set a single element at (point_idx, dim_idx).
    #[inline]
    pub fn set(&mut self, point_idx: usize, dim_idx: usize, value: f64) {
        self.0[(dim_idx, point_idx)] = value;
    }

    /// Get a point as a dynamically-sized vector.
    /// Efficient in column-major layout - returns a column directly.
    pub fn get_point(&self, point_idx: usize) -> DVector<f64> {
        self.0.column(point_idx).into()
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
    /// Very efficient in column-major layout.
    pub fn iter_points(&self) -> impl Iterator<Item = DVector<f64>> + '_ {
        self.0.column_iter().map(|col| col.into())
    }

    /// Filter points based on a boolean mask.
    pub fn filter_points(&self, mask: &[bool]) -> Self {
        assert_eq!(
            mask.len(),
            self.n_points(),
            "Mask length must match number of points"
        );

        let n_dims = self.n_dims();
        let mut flat = Vec::new();

        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                for d in 0..n_dims {
                    flat.push(self.get(i, d));
                }
            }
        }

        let n_kept = flat.len() / n_dims;
        Self::from_row_slice(n_kept, n_dims, &flat)
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

    #[allow(deprecated)]
    #[deprecated(since = "0.1.0", note = "Use direct methods on DataMatrix instead")]
    /// Borrow this matrix as a point view.
    pub fn as_points(&self) -> DataPoints<'_> {
        DataPoints { mat: &self.0 }
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

    /// Whether the view contains no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
