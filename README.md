# Inlier

`inlier` is a rust library for robust model fitting.
It exposes idiomatic Rust API with
pluggable estimators, samplers, scoring functions, and optimizers.

## Features

- Drop-in estimators for homography, fundamental/essential matrices, absolute pose,
  rigid transform, etc.
- Ready-to-use RANSAC variants (RANSAC, MSAC, MAGSAC, ACRANSAC) with local
  optimization pipelines (LSQ, IRLS, nested RANSAC, graph-cut-inspired smoothing, etc.).
- Flexible samplers (uniform, PROSAC, NAPSAC, progressive NAPSAC, adaptive
  reordering) plus neighborhood graphs (grid, USearch ANN).
- Optional `p3p` feature to enable Lambda-Twist minimal absolute pose solver.
- Optional `kornia-pnp` feature to enable EPnP/OnP absolute pose solving via `kornia-pnp`.
- Optional `graph-cut` feature enables a petgraph-backed k-NN Potts smoother local optimizer.
- Optional per-point priors (e.g., from a learned detector) via `MetasacSettings::point_priors` for weighted scoring.

### Run Examples

```bash
# Robust line fitting with plot output
cargo run --example linear_regression

# Homography estimation demo using real-image correspondences
cargo run --example homography_estimation

# Rigid transform estimation demo with optional alignment plot
cargo run --example rigid_transform

# Optional: fetch the real image pair from inlier-data and plot the scene
uv run --group dev python python/demo_homography_scene.py

# Optional: render the rigid-transform alignment plot
uv run --group dev python python/demo_rigid_transform_scene.py
```

See [`examples/README.md`](examples/README.md) for the full catalog.

Large binary example inputs are kept in the separate `inlier-data` repository so this repository stays focused on code and lightweight metadata. During development, clone `inlier-data`, push new binary assets there, and let example tooling fetch them through the `inlier_data` `pooch` interface when needed.

## Usage

```rust
use inlier::{estimate_homography, MetasacSettings};
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points1 = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let points2 = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]);

    let result = estimate_homography(&points1, &points2, 1.0, Some(MetasacSettings::default()))?;
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

This uses the c++ library [`superansac`](https://github.com/danini/superansac) as reference implementation, of which the author had done an excellent work on improving the SOTA sample consensus algorithm!
