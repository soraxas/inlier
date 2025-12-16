# Pipeline components

The pipeline exposed by `SuperRansac` is entirely driven by [RansacSettings](../../src/settings.rs)
and its enum members. Each stage maps to a concrete implementation defined in the Rust modules
listed below.

## Sampling

- **Uniform**: `UniformRandomSampler` randomly draws minimal samples. See
  [src/samplers/uniform.rs](../../src/samplers/uniform.rs).
- **PROSAC**: `ProsacSampler` uses inlier probability ordering to accelerate good samples.
  Implementation lives in [src/samplers/prosac.rs](../../src/samplers/prosac.rs).
- **NAPSAC / Progressive NAPSAC**: Spatial samplers that choose neighborhoods from
  [NeighborhoodGraph](../../src/samplers/neighborhood.rs) helpers (`Grid` or `Usearch`).
- **Importance / AR samplers**: Use adaptive reordering or custom importance weights.
  The logic spans [src/samplers/importance.rs](../../src/samplers/importance.rs) and
  [src/samplers/adaptive_reordering.rs](../../src/samplers/adaptive_reordering.rs).

Selecting a sampler is done via `SamplerType` in `RansacSettings`: each variant maps to the
matching sampler implementation. Pass the `settings.sampler` enum to core or to the Python
bindings so runtime configuration is easy.

## Neighborhood graphs

Spatial samplers depend on neighborhood graphs. The crate provides a `GridNeighborhoodGraph`
and a `UsearchNeighborhoodGraph` that uses the `usearch` crate for approximate nearest
neighbors. Both implement the same [NeighborhoodGraph](../../src/samplers/neighborhood.rs)
trait, so you can swap or extend the graph implementation.

Neighborhood graphs back samplers such as `NapsacSampler` and `ProgressiveNapsacSampler` and
are exposed through the `NeighborhoodType` enum in `RansacSettings`. The `Grid` variant
connects to a lightweight uniform grid, while the `BruteForce` and `Flann*` variants rely on
explicit neighbor lists (the latter currently use the `usearch` index as a stand-in for FLANN).

## Estimators and models

Built-in estimators live under [src/estimators](../../src/estimators/mod.rs) and cover:

1. **Homography** – 4/8-point estimation for image-to-image transforms.
2. **Fundamental matrix** – 7-/8-point solvers plus bundle-adjustment-like refinements.
3. **Essential matrix** – Nistér five-point solver with optional bundle adjustment tweaks.
4. **Absolute pose** – Lambda Twist (`p3p` feature) or DLT fallback plus `kornia-pnp` refinements.
5. **Rigid transform** – 3D-to-3D Procrustes alignment.
6. **Line** – simple line models for quick experiments.

`Estimator` implementations can return multiple solutions per sample (e.g., essential matrix) and
provide both minimal and non-minimal (`estimate_model_nonminimal`) interfaces so local
optimizers can reuse inliers.

## Scoring

Scores are produced by the `Scoring` trait; built-in implementations in
[src/scoring.rs](../../src/scoring.rs) include:

- `RansacInlierCountScoring` – the classic inlier counter.
- `MsacScoring` – truncated squared residual cost negated so higher is always better.
- `MinpranScoring` – per-point weighted inlier counting for MinPRAN-style branches.

All scorers store the same `Score` struct (inlier count + value) and can attach per-point
priors via their `with_priors` helpers. The sigma-consensus++ infrastructure builds
precomputed `sigma_lut` tables during `build.rs` to evaluate gamma functions cheaply
(see the Algorithms chapter for details).

## Configuration enums

`RansacSettings` centralizes control of the pipeline:

- `scoring`: choose `ScoringType` (`Magsac`, `Minpran`, `Acransac`, etc.).
- `sampler`: choose `SamplerType`.
- `neighborhood`: choose `NeighborhoodType` for spatial samplers.
- `local_optimization` / `final_optimization`: select `LocalOptimizationType` (more in the next
  chapter).
- `termination_criterion`: currently `TerminationType::Ransac`.
- `inlier_selector`: pick an inlier selector strategy, e.g., `SpacePartitioningRansac`.
- `point_priors`: optional per-point weights that many scoring functions consume.

See [src/settings.rs](../../src/settings.rs) for the full definition and defaults that match
the C++ implementation.
