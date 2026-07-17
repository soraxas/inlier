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

## Property Tests And Fuzzing

`proptest` is the default tool for structured algorithm invariants. Add or
extend properties in `tests/estimator_safety.rs` for finite output, scale
stability, valid-input round trips, and degenerate-input rejection.

For numerical algorithms, assert residuals or backward error rather than exact
elements when multiple valid floating-point answers exist. Use absolute error
near zero, relative error across scales, ULP bounds only for near-correctly
rounded scalar operations, and `approx` assertions where appropriate. Compare
small cases against an independent algorithm or high-precision oracle when a
closed-form expected result is unavailable.

Do not use uniform random floats as the only numerical input source. Maintain
a deliberate adversarial corpus: signed zero, subnormals, extreme finite
values, NaN/infinity rejection, cancellation, one-ULP neighbors, nearly
singular/rank-deficient matrices, clustered eigenvalues, and dimensions around
SIMD/block boundaries. Construct matrices with intended structure (SPD as
`M^T M + alpha I`, controlled condition numbers, known spectra) instead of
filtering arbitrary random matrices.

`cargo-fuzz` targets untrusted byte input and must assert semantic safety, not
only absence of panics. Run the bounded public API target with:

```bash
just fuzz-public-api 60
```

Keep fuzz corpora and artifacts out of git. Use `cargo mutants` as a targeted,
scheduled quality check for `src/core.rs`, `src/api.rs`, `src/estimators/`, and
`src/scoring.rs`; do not make the full mutation suite a pull-request gate.

### Practical Priority

Every pull request runs `cargo nextest run`; coverage is collected with
`cargo llvm-cov nextest`; add focused `proptest` cases for changed numerical or
domain logic.

Nightly CI should run targeted `cargo mutants --package inlier` and short
`cargo-fuzz` campaigns. Weekly CI should run the full mutation suite and
longer fuzzing campaigns. Run Miri when code introduces unsafe behavior or
subtle memory assumptions; use Loom for concurrent algorithms and Kani only
for small, critical bounded proofs.

Every PR runs deterministic regressions, nextest, focused Proptest properties,
and small differential tests. Nightly runs increase property-case counts and
exercise multiple feature/backend configurations. Schedule wall-clock Criterion
benchmarks on controlled hardware; benchmark sizes, shapes, layouts, data
conditioning, throughput, allocation, and backend selection. Use
Iai-Callgrind for instruction/cache regression signals when shared CI timing is
too noisy.

For runtime regressions, use Criterion (already used through CodSpeed) for
statistical microbenchmarks. Use Iai-Callgrind when instruction-level,
noise-resistant measurements are required in CI. These are performance tools,
not replacements for nextest, coverage, mutation testing, property tests, or
fuzzing.
