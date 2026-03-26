# Inlier Library Examples

This directory contains example programs demonstrating how to use the `inlier` library for robust estimation tasks.

## Workflow Inventory

The table below tracks user-facing workflows rather than only filenames. This is
the coverage inventory for the current public Rust API surface plus the primary
registration workflows documented by the project.

| Workflow | Primary API / entry point | Current example coverage | Visual path | Status |
| --- | --- | --- | --- | --- |
| Generic robust estimation pipeline | Custom estimator + RANSAC pipeline shape | `linear_regression.rs`, `pipeline_scratch_linear_regression.rs` | Optional numeric/text output | Covered |
| Homography estimation | `estimate_homography` | `homography_estimation.rs` | `python/demo_homography_scene.py` renders `examples/assets/homography/homography_matches.png` from `inlier-data` images | Covered |
| Fundamental matrix estimation | `estimate_fundamental_matrix` | `fundamental_matrix.rs` | No dedicated visual yet | Covered |
| Essential matrix estimation | `estimate_essential_matrix` | `essential_matrix.rs` | Numerical output only | Covered |
| Absolute pose estimation | `estimate_absolute_pose` | `absolute_pose.rs` | Numerical output only | Covered |
| 2D line estimation | `estimate_line` | `line_fitting.rs`, `linear_regression.rs`, `plot_line_fitting.rs` | `plot_line_fitting.rs` writes `examples/line_fitting_plot.png` | Covered |
| High-level rigid transform estimation | `estimate_rigid_transform` | `rigid_transform.rs` | `python/demo_rigid_transform_scene.py` renders `examples/assets/rigid_transform/rigid_transform_alignment.png` | Covered |
| High-level rigid point-cloud registration | `pcr::register_rigid`, `pcr_api.rs` | `pcr_api.rs`, `rigid_registration.rs`, `point_cloud_registration.rs` | Existing point-cloud visualizations live outside this README flow | Covered (grouped) |
| High-level non-rigid point-cloud registration | `pcr::register_nonrigid`, `pcr_api.rs` | `pcr_api.rs`, `nonrigid_registration.rs`, `rbf_scale_field.rs` | Existing point-cloud visualizations live outside this README flow | Covered (grouped) |

### Visual-workflow policy

- Use a visual showcase when an image, plot, rendered alignment, or similar
  artifact materially improves understanding.
- Keep Rust examples focused on estimation and report generation when that keeps
  the workflow simpler.
- Use companion Python scripts with headless `matplotlib` where an offscreen
  visual step is helpful.
- Store large binary visual inputs in `inlier-data`, then fetch them through the
  `inlier_data` `pooch` interface during development or regeneration.

### Grouped workflows

- The homography workflow is intentionally split: the Rust example performs the
  estimation and emits report files, while the Python script renders the visual
  scene from those outputs.
- The point-cloud registration workflows are intentionally grouped around
  high-level registration tasks rather than one example per helper function.
  `pcr_api.rs` covers the public API shape, while the other point-cloud examples
  demonstrate more specialized rigid or non-rigid variants.

## Examples

### 1. `linear_regression.rs`
Demonstrates the general RANSAC workflow. Shows how to use the library for robust estimation tasks.

**Run:**
```bash
cargo run --example linear_regression
```

### 2. `homography_estimation.rs`
Estimates a homography matrix from a real image pair using curated tentative correspondences.

**Run:**
```bash
cargo run --example homography_estimation
```

**What it does:**
- Loads curated correspondences from `examples/assets/homography/correspondences.tsv`
- Uses a real Sacre-Coeur image pair fetched by the companion Python visualizer from `inlier-data`
- Estimates the homography using RANSAC while rejecting intentionally wrong tentative matches
- Writes report files that can optionally be plotted with the companion Python visualization script

**Visualize:**
```bash
# Headless matplotlib rendering (uses Agg backend internally)
uv run --group dev python python/demo_homography_scene.py
```

**Workflow note:**
- This homography workflow is intentionally grouped into a Rust estimation step plus a Python visualization step.
- Large real-image inputs live in the separate `inlier-data` repository so the core `inlier` repo stays slim.
- During development, clone `inlier-data`, add/push new binary assets there, and let the Python visualization step consume them through the `inlier_data` `pooch` registry.

### 3. `fundamental_matrix.rs`
Estimates a fundamental matrix from point correspondences between two camera views.

**Run:**
```bash
cargo run --example fundamental_matrix
```

**What it does:**
- Simulates two camera views of 3D points on a plane
- Adds outliers to test robustness
- Estimates the fundamental matrix using RANSAC
- Verifies the rank-2 constraint

### 4. `essential_matrix.rs`
Estimates an essential matrix from calibrated point correspondences between two views.

**Run:**
```bash
cargo run --example essential_matrix
```

**What it does:**
- Simulates two calibrated camera views of random 3D points
- Adds tentative outlier correspondences
- Estimates the essential matrix with RANSAC
- Prints singular values and inlier recovery summary

### 5. `absolute_pose.rs`
Estimates camera pose from 3D world points and their 2D image observations.

**Run:**
```bash
cargo run --example absolute_pose
```

**What it does:**
- Simulates a calibrated camera observing 3D landmarks
- Injects a small set of wrong matches
- Estimates the camera rotation and translation with RANSAC
- Prints rotation and translation error against ground truth

### 6. `rigid_transform.rs`
Estimates a rigid transform from 3D-3D point correspondences and writes a small report for visualization.

**Run:**
```bash
cargo run --example rigid_transform
```

**What it does:**
- Generates synthetic 3D correspondences from a known rigid transform
- Adds outlier matches and mild noise
- Estimates rotation and translation with RANSAC
- Writes `examples/assets/rigid_transform/estimated_correspondence_report.tsv`
- Writes `examples/assets/rigid_transform/estimated_rigid_transform.txt`

**Visualize:**
```bash
uv run --group dev python python/demo_rigid_transform_scene.py
```

### 7. `line_fitting.rs`
Demonstrates robust line fitting (using homography as a workaround for demonstration).

**Run:**
```bash
cargo run --example line_fitting
```

### 8. `plot_line_fitting.rs`
Robust line fitting with visualization. Creates a plot showing inliers, outliers, and the estimated line.

**Run:**
```bash
cargo run --example plot_line_fitting
```

**Output:** Creates `examples/line_fitting_plot.png` showing:
- Green circles: Inliers found by RANSAC
- Red circles: Outliers
- Blue line: True line

## Custom Estimators

For tasks like line fitting, you would typically create a custom estimator implementing the `Estimator` trait. The examples show the general workflow using the built-in geometric estimators.
