# TODO / Plan: Composable Pipeline & Recipes

Goal: allow users to build pipelines from scratch (even with Python-defined components), while providing best-known recipes for homography/fundamental/essential that can be customized.

## Plan

1) **CorePipeline in Rust (non-breaking)**
   - Struct holds: `estimator`, `sampler`, `scoring`, `local_opt`, `final_opt`, `termination`, optional `preconditioner`.
   - `run(data)` executes the current linear RANSAC loop.
   - `with_*` methods to swap components.
   - Helper to build from settings defaults.

2) **Trait object adapters for Python**
   - PyO3 adapters for Sampler, Estimator, Scoring, LocalOptimizer, Termination, Preconditioner that call back into Python objects.
   - Ensure thread-safety or document single-threaded use.

3) **Expose builder to Python**
   - PyClass `Pipeline` (or `RansacPipeline`) wrapping the Rust builder.
   - Methods: `with_sampler`, `with_estimator`, `with_scoring`, `with_local_opt`, `with_final_opt`, `with_preconditioner`, `with_termination`, `run(data)`.
   - Accept built-in enums or Python adapters.

4) **Refactor existing `estimate_*_py` into recipes**
   - Homography/Fundamental/Essential helpers become thin recipes that build a pipeline with best defaults, then run.
   - Add `make_*_pipeline()` helpers returning a configurable pipeline object (so users can swap components before `run`).
   - Keep return shape backward compatible.

5) **Preconditioner hook**
   - Implement Hartley normalizer (2D–2D) and K-based normalizer (E/PnP) as built-ins; optional in builder/recipes.
   - Allow Python preconditioners via adapter.

6) **Keep recipes user-swappable**
   - Pipelines returned by recipes support `.with_*` before `.run` in Python.
   - Document defaults and override examples.

7) **Optional staged/batch runner (later)**
   - After linear path is stable, add `StagedPipeline` using same traits with staged scoring; expose as optional Python class.

## Rollout
- Implement builder + preconditioner trait + Python adapters/wrapper.
- Refactor one recipe (homography) as PoC; then extend to F/E.
- Add docs/examples showing: build-from-scratch with Python components; use recipe then swap sampler.
