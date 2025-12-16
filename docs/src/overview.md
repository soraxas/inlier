# Overview

`inlier` provides generic pipeline orchestrates
estimators, samplers, scoring rules, local optimizers, termination checks, and optional
inlier selectors. The orchestrator lives in [src/core.rs](../../src/core.rs) as
`SuperRansac`, and the traits it depends on are documented inline so you can drop in your own
models or samplers.

The high-level API in [`src/api.rs`](api) wires this generic pipeline into familiar functions
like `estimate_homography`, `estimate_fundamental_matrix`, and `estimate_absolute_pose`. Each
function builds the sampling/scoring/optimization stack over `RansacSettings` so you can
tune confidence, iteration limits, or probabilities without touching the low-level traits.

Key modules:

- [src/estimators](../../src/estimators/mod.rs) – built-in geometric estimators
  (homography, fundamental, essential, absolute pose, rigid transform, line).
- [src/samplers](../../src/samplers/mod.rs) – uniform, PROSAC, NAPSAC, progressive NAPSAC,
  AR, importance sampling, and neighborhood helpers.
- [src/scoring.rs](../../src/scoring.rs) – scoring primitives (RANSAC, MSAC, MinPRAN) plus
  the generated `sigma_lut` used by σ-consensus++.
- [src/optimisers](../../src/optimisers/mod.rs) – local refinement strategies plus
  reusable helpers.
- [python/inlier](../../src/python/mod.rs) – Python bindings built with PyO3 and
  distributed via `pip`/`npm`.
- [docs/src](..) – this mdBook holds the human-readable guide you are reading now.

See [src/settings.rs](../../src/settings.rs) for `RansacSettings` and the
enumerations that control all the components listed above.
