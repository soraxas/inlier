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
- Preprocessing helpers for voxel downsampling, normals, and FPFH.
- Multi-scale local refinement (coarse -> fine).
- Pluggable correspondence search and estimators.
- Colored point-to-plane kernel (Park2017-style).
- Geometry-only point-to-plane kernel.
- Robust loss weighting (Huber/Tukey) for outlier resistance.

This lets you swap kernels without changing the rest of the pipeline, and combine a
global initializer with a local refinement strategy.
