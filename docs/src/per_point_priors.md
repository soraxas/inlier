# Per-point priors and probabilities

[RansacSettings](../../src/settings.rs) exposes the optional
\\(\texttt{point\_priors}: \mathrm{Option}<\mathrm{Vec}<f64>>\\) field.
When supplied, the scoring modules consume the vector as weights aligned to the rows in your
data matrix. Each scoring implementation provides a `.with_priors(&priors)` helper before it is
passed into `SuperRansac`.

Examples:

```rust
let mut settings = RansacSettings::default();
settings.point_priors = Some(priors.clone());
settings.scoring = ScoringType::Msac;

let scoring = MsacScoring::new(settings.inlier_threshold, residual_fn)
    .with_priors(priors);
```

The weighting happens inside [src/scoring.rs](../../src/scoring.rs): both
`RansacInlierCountScoring` and `MsacScoring` look up the prior for each point and use it to
scale the support (inlier count) or the truncated squared cost.

Priors are also available inside the [src/api.rs](../../src/api.rs)
module; every `estimate_*` helper checks `settings.point_priors` and injects them before
running the pipeline. When your priors come from a neural network or sensor model, normalize them
externally (they behave like probabilities, but the scoring functions treat them as soft weights,
so values \\(> 1\\) are interpreted as trusted points).

Currently, only the scoring stage consumes the priors; samplers still operate on geometric
distributions, and local optimizers rely on residuals only. Adding a sampler-aware probability is
planned but not yet implemented.
