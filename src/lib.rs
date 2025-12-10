//! # Inlier - Robust Estimation with RANSAC
//!
//! `inlier` is a Rust port of the C++ SupeRANSAC library, providing robust estimation
//! algorithms for geometric models using RANSAC and its variants.
//!
//! ## Quick Start
//!
//! The easiest way to use `inlier` is through the high-level API functions:
//!
//! ```rust
//! use inlier::{estimate_homography, RansacSettings};
//! use nalgebra::DMatrix;
//!
//! // Create point correspondences
//! let points1 = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
//! let points2 = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]);
//!
//! // Estimate homography
//! let result = estimate_homography(&points1, &points2, 1.0, None).unwrap();
//! println!("Found {} inliers", result.inliers.len());
//! ```
//!
//! ## Extending the Library
//!
//! `inlier` is designed to be extensible. You can implement custom estimators, samplers,
//! scoring strategies, optimizers, and more by implementing the provided traits.
//!
//! ### Core Extension Traits
//!
//! The library is built around several key traits that you can implement:
//!
//! - **[`Estimator`](core::Estimator)**: Implement this to add support for new geometric models
//! - **[`Sampler`](core::Sampler)**: Implement this to create custom sampling strategies
//! - **[`Scoring<M>`](core::Scoring)**: Implement this to define custom scoring methods
//! - **[`LocalOptimizer<M, S>`](core::LocalOptimizer)**: Implement this for custom optimization strategies
//! - **[`TerminationCriterion<S>`](core::TerminationCriterion)**: Implement this for custom stopping criteria
//! - **[`InlierSelector<M>`](core::InlierSelector)**: Implement this for custom inlier pre-selection
//! - **[`NeighborhoodGraph`](samplers::NeighborhoodGraph)**: Implement this for spatial neighborhood structures
//!
//! ### Example: Custom Estimator
//!
//! ```rust
//! use inlier::core::Estimator;
//! use inlier::types::DataMatrix;
//!
//! // Define your model type
//! #[derive(Clone)]
//! struct MyModel {
//!     // Your model parameters
//! }
//!
//! // Implement the Estimator trait
//! struct MyEstimator;
//!
//! impl Estimator for MyEstimator {
//!     type Model = MyModel;
//!
//!     fn sample_size(&self) -> usize {
//!         3  // Minimum number of points needed
//!     }
//!
//!     fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
//!         // Check if the sample is geometrically valid
//!         sample.len() >= self.sample_size()
//!     }
//!
//!     fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
//!         // Estimate model from minimal sample
//!         vec![MyModel {}]
//!     }
//!
//!     fn is_valid_model(
//!         &self,
//!         model: &Self::Model,
//!         data: &DataMatrix,
//!         sample: &[usize],
//!         threshold: f64,
//!     ) -> bool {
//!         // Validate the estimated model
//!         true
//!     }
//! }
//! ```
//!
//! ### Example: Custom Sampler
//!
//! ```rust
//! use inlier::core::Sampler;
//! use inlier::types::DataMatrix;
//!
//! struct MySampler {
//!     // Your sampler state
//! }
//!
//! impl Sampler for MySampler {
//!     fn sample(
//!         &mut self,
//!         data: &DataMatrix,
//!         sample_size: usize,
//!         out_indices: &mut [usize],
//!     ) -> bool {
//!         // Implement your sampling strategy
//!         // Return true if successful, false otherwise
//!         true
//!     }
//!
//!     fn update(
//!         &mut self,
//!         sample: &[usize],
//!         sample_size: usize,
//!         iteration: usize,
//!         score_hint: f64,
//!     ) {
//!         // Update sampler state based on iteration results
//!     }
//! }
//! ```
//!
//! ### Example: Custom Scoring
//!
//! ```rust
//! use inlier::core::Scoring;
//! use inlier::types::DataMatrix;
//!
//! struct MyModel {
//!     // Your model
//! }
//!
//! #[derive(Clone, PartialOrd, PartialEq)]
//! struct MyScore(f64);
//!
//! struct MyScoring {
//!     threshold: f64,
//! }
//!
//! impl Scoring<MyModel> for MyScoring {
//!     type Score = MyScore;
//!
//!     fn threshold(&self) -> f64 {
//!         self.threshold
//!     }
//!
//!     fn score(
//!         &self,
//!         data: &DataMatrix,
//!         model: &MyModel,
//!         inliers_out: &mut Vec<usize>,
//!     ) -> Self::Score {
//!         // Compute score and populate inliers
//!         MyScore(0.0)
//!     }
//! }
//! ```
//!
//! ## Modules
//!
//! - **[`api`](api)**: High-level API functions for common estimation tasks
//! - **[`core`](core)**: Core traits and the main `SuperRansac` pipeline
//! - **[`estimators`](estimators)**: Built-in estimators for geometric models
//! - **[`samplers`](samplers)**: Built-in sampling strategies
//! - **[`scoring`](scoring)**: Built-in scoring strategies
//! - **[`models`](models)**: Geometric model types
//! - **[`settings`](settings)**: Configuration types for RANSAC pipelines
//! - **[`bundle_adjustment`](bundle_adjustment)**: Numerical optimization routines

pub mod api;
pub mod bundle_adjustment;
pub mod core;
pub mod estimators;
pub mod models;
pub mod nister_stewenius;
pub mod samplers;
pub mod scoring;
pub mod settings;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
pub mod python;

// Re-export high-level API
pub use api::{
    EstimationResult, estimate_absolute_pose, estimate_essential_matrix,
    estimate_fundamental_matrix, estimate_homography, estimate_line, estimate_rigid_transform,
};

// Re-export core traits for easy access
pub use core::{Estimator, InlierSelector, LocalOptimizer, Sampler, Scoring, TerminationCriterion};

// Re-export settings for convenience
pub use settings::RansacSettings;
