# Repository Guidelines

## Test and Benchmark Data

Keep generated fixtures and small, deterministic synthetic inputs in this repository.
Store large datasets, point clouds, images, and other heavyweight benchmark assets in the
`inlier-data` submodule. Do not add large binary data directly to this repository; add it to
`inlier-data` and reference it from tests, benchmarks, examples, or CI instead.

For release-asset, registry, fixture-packaging, and Pooch distribution details, follow
`inlier-data/AGENTS.md` in the data repository. Canonical benchmark inputs must remain
separate from visual-only or re-encoded image artifacts.

## CodSpeed Benchmarks

`benches/estimators.rs` is the deterministic public-API performance matrix used
by CodSpeed. It covers every public estimator, supported high-level scoring
mode, clean/noisy/outlier scenes, and the two sampler choices currently wired
through the public APIs: `Uniform` and `Prosac`.

`benches/components.rs` isolates all implemented sampler classes, Grid and
KD-tree neighborhood construction, and the identity preconditioner. Keep
component benchmarks deterministic and in-memory. Add a high-level matrix
dimension only when the public API actually honors it: current API functions
always use least-squares local/final optimization, and
`IdentityPreconditioner` is the only implemented preconditioner.

`benches/rejection.rs` tracks malformed/degenerate public inputs for every
estimator. Rejection benchmarks must assert an `Err` result and use a bounded,
deterministic iteration budget so they catch validation regressions without
turning the CodSpeed job into a stress test.

Run these locally with:

```bash
cargo bench --bench estimators -- --list
cargo bench --bench components -- --quick
cargo bench --bench rejection -- --quick
```
