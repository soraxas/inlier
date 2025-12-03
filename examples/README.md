# Inlier Library Examples

This directory contains example programs demonstrating how to use the `inlier` library for robust estimation tasks.

## Examples

### 1. `linear_regression.rs`
Demonstrates the general RANSAC workflow. Shows how to use the library for robust estimation tasks.

**Run:**
```bash
cargo run --example linear_regression
```

### 2. `homography_estimation.rs`
Estimates a homography matrix from 2D point correspondences with outliers.

**Run:**
```bash
cargo run --example homography_estimation
```

**What it does:**
- Generates synthetic 2D point correspondences with a known transformation
- Adds outliers to test robustness
- Estimates the homography using RANSAC
- Reports inlier count and estimated transformation

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

### 4. `line_fitting.rs`
Demonstrates robust line fitting (using homography as a workaround for demonstration).

**Run:**
```bash
cargo run --example line_fitting
```

### 5. `plot_line_fitting.rs`
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
