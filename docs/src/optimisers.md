# Optimisers

Optimisation strategies implement `crate::optimisers::LocalOptimizer` defined in
[src/optimisers/mod.rs](../../src/optimisers/mod.rs) and the concrete variants live in
[src/optimisers/local.rs](../../src/optimisers/local.rs). These optimizers can be attached as
`RansacSettings.local_optimization` or `final_optimization` values via `LocalOptimizationType`.

Key optimizers:

- `LeastSquaresOptimizer` – simple weighted refit on all inliers.
- `IteratedLeastSquaresOptimizer` – repeated refits for convergence.
- `NestedRansacOptimizer` – inner RANSAC loop sampling subsets of inliers again.
- `IRLSOptimizer` – iteratively reweighted least squares with MSAC-style weights.
- `CrossValidationOptimizer` – bootstrap resampling builds per-point confidence weights.
- `GraphCutLocalOptimizer` (requires `graph-cut` feature) – builds a k-NN graph and runs label flips to promote spatial coherence.

```rust
struct MyOptimizer;
impl inlier::optimisers::LocalOptimizer<MyModel, MyScore> for MyOptimizer {
    fn run(... ) -> (...) { /* custom refinement */ }
}
```

The module also documents the `NoopLocalOptimizer` used when no refinement is desired. Use the
`mdbook-preprocessor-include` macro to embed snippets from the real implementation so docs stay
synchronized:
