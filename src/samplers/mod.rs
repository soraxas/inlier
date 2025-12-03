//! Sampling strategies for SupeRANSAC.
//!
//! This module contains the sampling strategies used by SupeRANSAC.
//!
//! The goal is to mirror the behavior of the original C++ samplers found in
//! `superansac_c++/include/samplers`, while providing a small, idiomatic Rust
//! surface over the shared `Sampler` trait.

pub mod adaptive_reordering;
pub mod importance;
pub mod napsac;
pub mod neighborhood;
pub mod progressive_napsac;
pub mod prosac;
pub mod uniform;

// Re-export all samplers and traits for convenience
pub use adaptive_reordering::AdaptiveReorderingSampler;
pub use importance::ImportanceSampler;
pub use napsac::NapsacSampler;
pub use neighborhood::{
    DummyNeighborhood, GridNeighborhoodGraph, NeighborhoodGraph, UsearchNeighborhoodGraph,
};
pub use progressive_napsac::ProgressiveNapsacSampler;
pub use prosac::ProsacSampler;
pub use uniform::UniformRandomSampler;
