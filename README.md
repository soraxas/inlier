# Inlier

`inlier` is a rust library for robust model fitting.
It exposes idiomatic Rust API with
pluggable estimators, samplers, scoring functions, and optimizers.

## Features

- Drop-in estimators for homography, fundamental/essential matrices, absolute pose,
  rigid transform, etc.
- Ready-to-use RANSAC variants (RANSAC, MSAC, MAGSAC, ACRANSAC) with local
  optimization pipelines (LSQ, IRLS, nested RANSAC, etc.).
- Flexible samplers (uniform, PROSAC, NAPSAC, progressive NAPSAC, adaptive
  reordering) plus neighborhood graphs (grid, USearch ANN).

### Run Examples

```bash
# Robust line fitting with plot output
cargo run --example linear_regression

# Homography estimation demo
cargo run --example homography_estimation
```

See [`examples/README.md`](examples/README.md) for the full catalog.

## Usage

```rust
use inlier::{estimate_homography, RansacSettings};
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points1 = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let points2 = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]);

    let result = estimate_homography(&points1, &points2, 1.0, Some(RansacSettings::default()))?;
    println!("Inliers: {}", result.inliers.len());
    Ok(())
}
```

## Extensibility

Implement the core traits to add custom functionality:

- `Estimator` (models)
- `Sampler` / `NeighborhoodGraph`
- `Scoring`
- `LocalOptimizer`
- `TerminationCriterion`

Each trait includes documentation and doctests describing the expected behavior.

## Reference

This uses the c++ library `superansac` as reference implementation, who had done an excellent work on improving the SOTA sample consensus algorithm.
