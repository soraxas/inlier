# Algorithm reference

This chapter records algorithmic decisions borrowed from the SupeRANSAC / MAGSAC++ papers
that the Rust port implements.

## \\(\sigma\\)-consensus++ weights

The new \\(\sigma\\)-consensus++ (MAGSAC++) quality function comes from the iteratively reweighted least
squares formulation explained in the paper: every point receives a weight
\\[
w(r) = \int g(r \mid \sigma) f(\sigma) \, d\sigma,
\\]
 where the density \\(g\\) depends on the
\\(\chi\\)-distribution and the maximum noise scale \\(\sigma_{\text{max\\).

Rather than approximating the integrals at runtime, this crate precomputes the incomplete
gamma functions for common \\((\text{dof}, k)\\) pairs. The gamma LUT is generated in [build.rs](../../build.rs)
using the Lanczos approximation:

```rust
const COEFFS: [f64; 9] = [
    676.520368..., -1259.139216..., 771.323428..., // truncated here for brevity
];

// Actual code walks the coefficients and builds a precalculated table.
```

`COEFFS` are the standard Lanczos constants used when evaluating
\\[
    \ln \Gamma(x) \text{ for } x > 0.
\\]
[build.rs](../../build.rs) computes both the upper and lower incomplete gamma functions and stores them in a
generated `sigma_lut.rs` file inside `OUT_DIR`. This file is included into
[src/scoring.rs](../../src/scoring.rs) and cached behind a `OnceLock<Mutex<_>>`, so each
scoring invocation simply reads precomputed values instead of evaluating costly integrals.

## Score type and loss

The `Score` object in [src/scoring.rs](../../src/scoring.rs) mirrors the original design: it
stores the raw inlier count and the derived value (negative cost when using MSAC-style scoring).
The `RansacInlierCountScoring`, `MsacScoring`, and `MinpranScoring` implementations reuse the
same `threshold`, optional priors, and residual callback, ensuring the loss comparisons are
consistent as they bubble up into `SuperRansac`.

## Weight computation and LUT usage

Weights inside `IRLSOptimizer` are computed with a simple MSAC-style formula
\\[
    w = 1 - \frac{r^2}{\tau^2}
\\]
for inliers and zero for outliers. Still, the same LUT used for
\\(\sigma\\)-consensus++ can be plugged in later if you want to replace the simplified MSAC weights with the
paper's exact marginalization. The `sigma_lut` generator ships with several \\((\text{dof}, k)\\)
configurations that cover the most-common residual spaces (2D reprojection, 3D point-to-point, etc.) – just
re-run `cargo build` if you adjust the list inside [build.rs](../../build.rs).

For more context on the pipeline maths, refer to the original MAGSAC++ paper and the
[superansac_c++/](../../superansac_c++/) C++ reference.
