use crate::types::DataMatrix;

/// Local optimization strategy, potentially refining a model using its inliers.
///
/// After RANSAC finds a good model, local optimization can refine the current
/// hypothesis by re-estimating the model from all inliers or subsets thereof.
///
/// ## Example: custom refinement
///
/// ```rust
/// use inlier::optimisers::LocalOptimizer;
/// use inlier::types::DataMatrix;
///
/// #[derive(Clone)]
/// struct LineModel(f64, f64);
///
/// #[derive(Clone)]
/// struct LineScore(f64);
///
/// struct AveragingOptimizer;
///
/// impl LocalOptimizer<LineModel, LineScore> for AveragingOptimizer {
///     fn run(
///         &mut self,
///         _data: &DataMatrix,
///         inliers: &[usize],
///         model: &LineModel,
///         score: &LineScore,
///     ) -> (LineModel, LineScore, Vec<usize>) {
///         // A dummy optimizer that just returns the inputs.
///         (model.clone(), score.clone(), inliers.to_vec())
///     }
/// }
/// ```
pub trait LocalOptimizer<M, S: Clone> {
    /// Run local optimization on the current model and inliers.
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &M,
        best_score: &S,
    ) -> (M, S, Vec<usize>);
}

/// Local optimizer stub used when no refinement is desired.
pub struct NoopLocalOptimizer;

impl<M: Clone, S: Clone> LocalOptimizer<M, S> for NoopLocalOptimizer {
    fn run(
        &mut self,
        _data: &DataMatrix,
        inliers: &[usize],
        model: &M,
        best_score: &S,
    ) -> (M, S, Vec<usize>) {
        (model.clone(), best_score.clone(), inliers.to_vec())
    }
}

/// Concrete optimizer implementations that live in their own module to keep the
/// trait definition lean.
pub mod local;

pub use local::{
    CrossValidationOptimizer, IRLSOptimizer, IteratedLeastSquaresOptimizer, LeastSquaresOptimizer,
    NestedRansacOptimizer,
};

#[cfg(feature = "graph-cut")]
pub use local::GraphCutLocalOptimizer;
