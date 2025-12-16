# Samplers

`RansacSettings.sampler` selects one of the samplers in [src/samplers/mod.rs](../../src/samplers/mod.rs).
Most other samplers rely on neighborhood graphs from
[src/samplers/neighborhood.rs](../../src/samplers/neighborhood.rs) and optional priors.

The most common samplers:

1. `UniformRandomSampler` – draws without replacement; cheap and deterministic for debugging.
2. `ProsacSampler` – orders correspondences by score or prior and gradually considers more points.
3. `NapsacSampler` / `ProgressiveNapsacSampler` – sample neighborhoods by recursively splitting the image domain.
4. `ImportanceSampler` / `ArSampler` – reorders points based on running weights or estimated variances.

Each sampler implements `crate::core::Sampler` and exposes `sample` / `update` hooks. You can add
custom samplers by implementing the trait and plugging it into the runtime pipeline or the Python
adapter (see the `SamplerChoice` enum in [src/choices.rs](../../src/choices.rs)).

```rust
let mut sampler = inlier::samplers::UniformRandomSampler::new();
sampler.sample(&data_matrix, estimator.sample_size(), &mut indices);
sampler.update(&indices, estimator.sample_size(), iteration, 1.0);
```

Neighborhood graphs live under the same module and include:

- `GridNeighborhoodGraph` – simple spatial binning for image coordinates.
- `UsearchNeighborhoodGraph` – uses `usearch` for approximate nearest neighbors (Open-source FLANN replacement).

Use the `mdbook-preprocessor-include` plugin to keep sampler documentation in sync with the real code:
