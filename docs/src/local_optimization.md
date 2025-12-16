# Local optimization strategies

Local optimization is optional in `SuperRansac` and kicks in after a promising model is found.
Each optimizer implements the [LocalOptimizer trait](../../src/optimisers/mod.rs) and can
be swapped via the settings enums.
Each optimizer implements the `LocalOptimizer` trait in [`src/optimisers/mod.rs`](../../src/optimisers/mod.rs)
and can be plugged into both the Rust API and the Python bindings.

The concrete implementations live in [src/optimisers/local.rs](../../src/optimisers/local.rs):

- `LeastSquaresOptimizer` – refits using all inliers via `estimate_model_nonminimal`. It is the
  simplest optimizer and is used when `LocalOptimizationType::Lsq` is set.
- `IteratedLeastSquaresOptimizer` – repeatedly refits the model to encourage convergence. It is a
  lightweight loop around the non-minimal estimator and respects a maximum number of iterations.
- `NestedRansacOptimizer` – runs a secondary RANSAC loop inside the inliers. It samples subsets of
  the current inliers and refines the best model discovered inside this loop (`LocalOptimizationType::NestedRansac`).
- `IRLSOptimizer` – Iteratively Reweighted Least Squares with MSAC-inspired weights. The
  `residual_fn` generates weights and a refit is done until convergence or a max iteration count.
- `CrossValidationOptimizer` – bootstrap sampling of inliers to compute soft weights, then refit
  to produced weights; useful when you want a Monte Carlo estimate of point reliability.
- `GraphCutLocalOptimizer` (behind the `graph-cut` feature) – builds a k-NN graph with `petgraph`
  and performs label flips with Potts pairwise terms. After convergence, it refits only the
  nodes labeled as inliers.

Every optimizer is transparent about its configuration (max iterations, weight usage, etc.), and
they all return the refined model, score, and inliers tuple expected by `SuperRansac`.

The enum `LocalOptimizationType` in [src/settings.rs](../../src/settings.rs) selects the optimizer,
and `RansacSettings.local_optimization_settings` exposes parameters such as `max_iterations`,
`graph_cut_number`, and `spatial_coherence_weight`. The `final_optimization` field lets you run a
different optimizer once the outer loop terminates.

For GraphCut, the optimizer builds a graph (default k=8 neighbors) and charges a smoothness
penalty when neighbors disagree. After label stabilization, `estimate_model_nonminimal` is called
on the refined inlier set, so the actual refit still lives in your estimator.

Reference the source for details and act as a template for writing your own optimizer:

```
// Implement the LocalOptimizer trait in your crate if you want a custom refinement step.
// See [`src/optimisers/local.rs`](../../src/optimisers/local.rs) for examples.
```
