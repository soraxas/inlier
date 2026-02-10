//! KISS-Matcher implementation for point cloud registration
//!
//! This module implements techniques from KISS-Matcher:
//! - FasterPFH (Fast Point Feature Histogram) descriptors
//! - Feature matching with mutual NN and ratio test
//! - ROBIN matching (Robust Bilateral Network) with k-core pruning
//! - GNC solver for robust rotation/translation estimation
//!
//! Reference: "KISS-Matcher: A Simple Yet Effective 3D Point Cloud Registration Method"

pub mod config;
pub mod correspondence;
pub mod features;
pub mod gnc;
pub mod matching;
//pub mod pipeline; // Old simple pipeline - temporarily disabled
pub mod pipeline_full;
pub mod pipeline_nonrigid;
pub mod sipfh;

pub use config::KISSMatcherConfig;
pub use correspondence::{Correspondence, FeatureMatcher};
pub use gnc::{GNCResult, GNCSolver};
pub use pipeline_full::{KISSMatcherFullResult, kiss_matcher_full_pipeline};
pub use pipeline_nonrigid::{
    NonRigidKISSConfig, NonRigidKISSResult, nonrigid_kiss_matcher_pipeline,
};
pub use sipfh::{SIPFH, SIPFHConfig, SIPFHFeaturePoint, ScaleKeypoint};
