# Scoring

The scoring stage implements `crate::core::Scoring` and lives in [src/scoring.rs](../../src/scoring.rs).
It defines the shared `Score` struct plus several scoring strategies:

- `RansacInlierCountScoring` – classic RANSAC; counts points whose residuals sit below `threshold`.
- `MsacScoring` – truncated squared cost turned into a maximization objective via `value = -cost`.
- `MinpranScoring` – per-point weighted version inspired by MinPRAN.

Each scorer optionally accepts `point_priors` via `.with_priors(&priors)` and reuses a common residual
callback. The module also includes `include!(concat!(env!("OUT_DIR"), "/sigma_lut.rs"))`, so the
sigma-consensus++ infrastructure runs from precomputed incomplete gamma tables generated in
[build.rs](../../build.rs).

```rust
let scoring = inlier::scoring::MsacScoring::new(1.5, residual_fn);
let mut inliers = Vec::new();
let score = scoring.score(&data_matrix, &model, &mut inliers);
assert!(score.value.is_sign_negative());
```
