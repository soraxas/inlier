# Point cloud registration guide

Colored ICP is a strong local refinement method, but it is only one option. For VGGT-like
reconstructions (inferred point clouds with RGB), the quality of geometry and color varies
across frames, so a practical pipeline should separate global alignment from local refinement.

## Recommended pipeline

- **Preprocess**: voxel downsample -> optional outlier removal -> estimate normals.
- **Global init (if pose is weak)**: FPFH + RANSAC or TEASER++ to get a coarse transform.
- **Local refine**: point-to-plane ICP or GICP with a robust loss.
- **Multi-scale**: coarse-to-fine voxel scales with tighter thresholds at each level.

## When to use colored ICP

Use colored ICP when:
- you have reliable RGB and texture variation,
- geometry alone allows tangential slipping (planar scenes),
- you already have a decent initial pose.

Avoid relying on the color term when:
- colors are inconsistent across frames,
- the scene is textureless or heavily specular,
- your RGB comes from hallucinated or noisy colorization.

## Alternatives worth considering

- **Point-to-plane ICP**: fast, stable when normals are reliable.
- **GICP**: strong on structured scenes and mixed planar/edge geometry.
- **Robust ICP**: Huber/Tukey/trimmed losses help with partial overlap or noisy scans.
- **NDT**: good basin of convergence on sparse or noisy point clouds.

## Support in this crate

The point cloud module provides both global registration and ICP-style refinement:

- Global registration with FPFH feature matching and RANSAC.
- Fast global registration (FGR) from feature matches or explicit correspondences.
- Correspondence checkers (edge length / distance / normal) to prune bad hypotheses.
- Preprocessing helpers for voxel downsampling, normals, and FPFH.
- Multi-scale local refinement (coarse -> fine).
- Pluggable correspondence search and estimators.
- Colored point-to-plane kernel (Park2017-style).
- Geometry-only point-to-plane kernel.
- Robust loss weighting (Huber/Tukey) for outlier resistance.

This lets you swap kernels without changing the rest of the pipeline, and combine a
global initializer with a local refinement strategy.

## Auto-tuning heuristics

The crate includes `auto_tune_pipeline()` which picks voxel size and radii from the
median nearest-neighbor spacing, following common heuristics used by Open3D:

- `voxel_size ~= 2.5x` median spacing
- `normal_radius ~= 2x` voxel size
- `fpfh_radius ~= 5x` voxel size
- `global_distance_threshold ~= 1.5x` voxel size

These rules are the same style of guidance used in the global registration tutorial
and the original FPFH paper.

References:
- Choi, Zhou, Koltun. "Robust Reconstruction of Indoor Scenes" (2015).
- Rusu, Blodow, Beetz. "Fast Point Feature Histograms (FPFH) for 3D Registration" (2009).
- Zhou, Park, Koltun. "Fast Global Registration" (2016).

## Global registration examples

RANSAC + ICP:

```rust
use inlier::{
    auto_tune_pipeline, GlobalRegistrationMethod, PointCloud,
};

let mut pipeline = auto_tune_pipeline(&source, &target)?;
pipeline.global_method = GlobalRegistrationMethod::Ransac;
let result = pipeline.run(&source, &target)?;
```

Fast global registration (FGR) + ICP:

```rust
use inlier::{
    auto_tune_pipeline, GlobalRegistrationMethod, PointCloud,
};

let mut pipeline = auto_tune_pipeline(&source, &target)?;
pipeline.global_method = GlobalRegistrationMethod::FastGlobalRegistration;
let result = pipeline.run(&source, &target)?;
```

Correspondence-based global registration:

```rust
use inlier::{
    registration_fgr_based_on_correspondence, registration_ransac_based_on_correspondence,
    FastGlobalRegistrationOptions, GlobalRegistrationSettings,
};

let correspondences: Vec<(usize, usize)> = /* your matches */;
let ransac = GlobalRegistrationSettings::default();
let _ = registration_ransac_based_on_correspondence(
    &source_down, &target_down, &correspondences, &ransac,
)?;

let fgr = FastGlobalRegistrationOptions::default();
let _ = registration_fgr_based_on_correspondence(
    &source_down, &target_down, &correspondences, &fgr,
)?;
```

## Absolute pose notes

For real-world 2D-3D correspondences, normalize image coordinates with intrinsics:

```
x = (u - cx) / fx
y = (v - cy) / fy
```

The `AbsolutePoseEstimator` uses a minimal 3-point solver. The default build falls back to
a simplified DLT, which is often too weak for real data. Enable the optional `p3p` feature
to use a proper P3P solver:

```
cargo run --example absolute_pose_real --features p3p
```

For stronger refinement on 4+ points, enable `kornia-pnp`:

```
cargo run --example absolute_pose_real --features "p3p kornia-pnp"
```
