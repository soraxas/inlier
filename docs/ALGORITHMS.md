# Algorithms Overview

This crate mirrors the SupeRANSAC architecture with pluggable pieces. This doc summarizes the main components and the graph-cut-inspired local optimizer.

## Pipeline
- **Sampling:** Uniform, PROSAC, NAPSAC, progressive NAPSAC, adaptive reordering. Neighborhoods use grid or USearch ANN.
- **Estimators:** Homography (4/8 point), Fundamental (7/8 point with bundle adjustment), Essential (5-point Nister with bundle adjustment), Absolute pose (Lambda-Twist P3P when `p3p` feature is enabled, DLT fallback otherwise, EOnP via `kornia-pnp` feature), Rigid transform (Procrustes).
- **Scoring:** RANSAC and MSAC are exact; MAGSAC/ACRANSAC approximations; scores expose per-point residual evaluation but do not yet consume external per-point priors.
- **Local optimization:** LSQ, Iterated LSQ, IRLS (MSAC weights), Nested RANSAC, Cross-validation, and the graph-cut smoother (feature `graph-cut`).

## Graph-Cut Local Optimizer (feature `graph-cut`)
- **Inputs:** Current model, data matrix, inlier set.
- **Graph:** Build an undirected k-NN graph (default k=8) over all points using Euclidean distance in the data rows; edges carry the distance as a weight.
- **Unary term:** Residual r_i = residual_fn(data, model, i). Cost for labeling point i as inlier is r_i / threshold; outlier cost is a small constant.
- **Pairwise term:** Potts penalty `smooth_weight` when neighboring labels disagree; encourages spatially smooth inlier regions.
- **Update:** Iterate label updates (inlier/outlier) until stable or `max_iterations`; then refit the model with `estimate_model_nonminimal` on the refined inliers.
- **Nature:** Approximate graph-cut (no max-flow); fast and dependency-light. Intended as a smoother on top of standard scoring, not a replacement for full Graph-Cut RANSAC.

## Per-Point Priors / Probabilities
- Provide per-point priors via `RansacSettings.point_priors` (length = #rows in data). RANSAC/MSAC/MAGSAC scoring will weight costs and support accordingly.
- Samplers currently ignore priors (PROSAC uses ordering only); local optimizers still rely on residual-derived weights.
- Neighborhood samplers (PROSAC/adaptive reordering) use ordering or update rules but not explicit probability inputs.
- If per-point priors are needed, they could be threaded through scoring functions and local optimizers as weights; this is not yet implemented.
