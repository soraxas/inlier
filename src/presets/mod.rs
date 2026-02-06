//! High-level Rust API for SupeRANSAC.
//!
//! This module provides user-friendly functions for estimating geometric models
//! similar to the Python API.

pub mod rigid_registration;
pub mod rigid_transform;
pub mod similarity_registration;

pub use rigid_registration::{TeaserLikeRigidEstimator, rigid_registration_pipeline};
pub use rigid_transform::rigid_transform_pipeline;
pub use similarity_registration::similarity_registration_pipeline;
pub mod pointcloud_registration;
