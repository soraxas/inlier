# Repository Guidelines

## Test and Benchmark Data

Keep generated fixtures and small, deterministic synthetic inputs in this repository.
Store large datasets, point clouds, images, and other heavyweight benchmark assets in the
`inlier-data` submodule. Do not add large binary data directly to this repository; add it to
`inlier-data` and reference it from tests, benchmarks, examples, or CI instead.

For release-asset, registry, fixture-packaging, and Pooch distribution details, follow
`inlier-data/AGENTS.md` in the data repository. Canonical benchmark inputs must remain
separate from visual-only or re-encoded image artifacts.
