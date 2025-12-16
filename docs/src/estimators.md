# Estimators

The estimators bundled with `inlier` mirror the geometric models supported by SupeRANSAC:
homography, fundamental matrix, essential matrix, absolute pose, rigid transform, and line
estimation. Each estimator implements [crate::core::Estimator](../../src/core.rs) with a
`sample_size` suited to its minimal solver and exposes both minimal and non-minimal fitting entry
points.

Sections of interest:

- [src/estimators/homography.rs](../../src/estimators/homography.rs) – 4/8-point homography
  solvers plus normalization helpers.
- [src/estimators/fundamental.rs](../../src/estimators/fundamental.rs) – 7/8-point solvers with
  validity checks for the rank-2 constraint.
- [src/estimators/essential.rs](../../src/estimators/essential.rs) – Nistér five-point solver
  coupled with optional bundle-adjustment-style refinement.
- [src/estimators/absolute_pose.rs](../../src/estimators/absolute_pose.rs) – absolute pose using
  Lambda Twist when the `p3p` feature is enabled plus fallback DLT estimators.
- [src/estimators/rigid_transform.rs](../../src/estimators/rigid_transform.rs) – Procrustes alignment.

You can copy any estimator and plug it into the pipeline by implementing (see
[Estimator trait](../../src/core.rs) for details):

```rust
struct MyEstimator;
impl inlier::core::Estimator for MyEstimator {
    type Model = MyModel;
    fn sample_size(&self) -> usize { 2 }
    fn is_valid_sample(&self, data, sample) -> bool { true }
    fn estimate_model(&self, data, sample) -> Vec<Self::Model> { vec![MyModel::default()] }
}
```

This keeps the pipeline fully generic while the Python bindings can still expose it through
runtime enums or adapters.
