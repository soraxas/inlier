//! Core shared types for the SupeRANSAC Rust port.
//!
//! The goal here is to mirror the conceptual role of the C++ `DataMatrix`
//! and related aliases, without prematurely committing to a specific memory
//! layout or set of helpers. We start with simple `nalgebra` aliases that can
//! be extended or optimized later.

use nalgebra::DMatrix;

/// Dynamic matrix of `f64`, conceptually equivalent to the C++ `DataMatrix`
/// alias (`Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>`).
///
/// Many estimators and samplers in the original code operate on this type.
pub type DataMatrix = DMatrix<f64>;
