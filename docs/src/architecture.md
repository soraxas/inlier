# Architecture Overview

This file summarizes how to structure the robust estimation pipeline to remain modular today (linear RANSAC loop) while being ready for a staged/batched executor later. It focuses on composable interfaces and a builder-style API; recipes are thin wrappers over the same components.

## Core traits (linear pipeline)

The pipeline is a linear sequence: preprocess (optional) → sampler → estimator → scoring → (optional LO/FO) → termination.

```rust
pub trait Sampler {
    fn sample(&mut self, n_rows: usize, sample_size: usize, out: &mut [usize]) -> bool;
    fn update(&mut self, _sample: &[usize], _iter: usize, _score_hint: f64) {}
}

pub trait Estimator {
    type Model: Clone;
    fn sample_size(&self) -> usize;
    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model>;
    fn estimate_model_nonminimal(&self, _data: &DataMatrix, _sample: &[usize], _w: Option<&[f64]>) -> Vec<Self::Model> { vec![] }
}

pub trait Scoring<M> {
    type Score: PartialOrd + Clone;
    fn threshold(&self) -> f64;
    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score;
}

pub trait LocalOptimizer<M, S: PartialOrd> {
    fn optimize(&self, data: &DataMatrix, model: &M, inliers: &[usize]) -> (M, S, Vec<usize>);
}

pub trait TerminationCriterion {
    fn should_stop(&self, iteration: usize, best_inlier_ratio: f64) -> bool;
}
```

Optional wrapper for normalization/preconditioning:

```rust
pub trait Preconditioner<M> {
    fn normalize(&self, data: &DataMatrix) -> (DataMatrix, Normalization);
    fn denormalize(&self, model: &M, norm: &Normalization) -> M;
}
pub struct Normalization { /* e.g., transforms, scales */ }
```

## Linear executor (current SuperRansac)

Expose a builder so callers can override components and optionally wrap an estimator with a preconditioner.

```rust
pub struct RansacBuilder<E, Sa, Sc, LO, T> {
    estimator: E,
    sampler: Sa,
    scoring: Sc,
    local_opt: Option<LO>,
    final_opt: Option<LO>,
    termination: T,
    precond: Option<Box<dyn Preconditioner<E::Model>>>,
}

impl<E, Sa, Sc, LO, T> RansacBuilder<E, Sa, Sc, LO, T>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion,
{
    pub fn new(estimator: E, sampler: Sa, scoring: Sc, termination: T) -> Self { /* … */ }
    pub fn with_local_opt(mut self, lo: LO) -> Self { /* … */ }
    pub fn with_final_opt(mut self, fo: LO) -> Self { /* … */ }
    pub fn with_preconditioner(mut self, p: impl Preconditioner<E::Model> + 'static) -> Self { /* … */ }
    pub fn run(&mut self, data: &DataMatrix) -> Option<EstimationResult<E::Model>> { /* linear loop */ }
}
```

Recipes (homography, F, E, PnP, rigid) are just pre-filled builders.

## Staged/batched executor (DAG-in-a-loop, future-ready)

Add a second executor that reuses the same traits but introduces staged evaluation and batching. The “DAG” is per-epoch and acyclic; the outer loop carries state (best model, priors, subsets).

```rust
pub struct StagedEvaluator<M, S> {
    subsets: Vec<Vec<usize>>, // P1, P2, P3, P_all
    q_max: f64,
    _marker: PhantomData<(M,S)>,
}

impl<M, S: PartialOrd + Clone> StagedEvaluator<M, S> {
    fn score_with_bounds(
        &self,
        data: &DataMatrix,
        model: &M,
        scoring: &impl Scoring<M, Score=S>,
        best_score: &S,
    ) -> Option<(S, Vec<usize>)> { /* early-exit bound logic */ }
}

pub struct BatchParams { pub batch_size: usize, pub keep_top: usize, pub lo_top: usize; /* etc */ }

pub struct RansacBatchRunner<E, Sa, Sc, LO, T> { /* holds builder + staged evaluator + priors */ }

impl<E, Sa, Sc, LO, T> RansacBatchRunner<E, Sa, Sc, LO, T>
where
    E: Estimator,
    Sa: Sampler,
    Sc: Scoring<E::Model>,
    LO: LocalOptimizer<E::Model, Sc::Score>,
    T: TerminationCriterion,
{
    pub fn run_epoch(&mut self, data: &DataMatrix) -> Option<EstimationResult<E::Model>> {
        // 1) optional precond → data_n
        // 2) batch sample + sample degeneracy checks
        // 3) minimal solve (+ multi-solution expansion)
        // 4) cheap model sanity
        // 5) staged scoring with optimistic bounds (priority queue)
        // 6) LO/final refine on top-K
        // 7) update priors p_i, thresholds, termination state
    }
}
```

Priors/informed sampling: extend `Sampler` to optionally accept a weight vector (`Option<&[f64]>`) and ignore it by default; informed samplers can consume it. Carry `p_i: Vec<f64>` in the batch runner; update from residuals each epoch.

## Python bindings

- Keep recipe functions for simplicity.
- Add a builder-like object (or structured config) mirroring the Rust builder; later expose staged runner config (batch sizes, subsets, priors).
- Always thread a seed through settings to make samplers deterministic when requested.

## Learnable / R&D components

The same interfaces support learned or experimental pieces:

- **Learnable sampler**: implement `Sampler` that consumes priors/weights (e.g., from the batch runner’s `p_i`) and updates its internal policy from residuals.
- **Learnable estimator**: wrap a learned minimal solver behind `Estimator`; nonminimal fits can also call into learned modules.
- **Learnable scoring**: implement `Scoring` that maps residual features to scores; keep a bound on per-point contribution to preserve early-exit logic.
- **Experimental LO/termination**: add new implementations of existing traits without changing executors.

Because the execution model is linear per hypothesis and components are swappable, learned variants can be introduced incrementally, and exposed in Python as alternative samplers/estimators/scorers for R&D without altering the core loop.

## Parallel scoring with early abort (non-greedy parallelism)

To exploit parallelism without greediness:
- Score hypotheses in stages (P1/P2/…); after each stage compute an optimistic bound `UB = s(k) + (n−k)·q_max`.
- Keep the best score in an atomic `S_best`; if `UB <= S_best`, abort scoring that hypothesis.
- When a task finishes all stages and improves `S_best`, atomically publish the new best model/inliers; other tasks will see the tighter bound and self-prune.
- This allows parallel evaluation while preserving the usual RANSAC confidence/iteration criteria and avoids wasted work once a better incumbent appears.

## Execution model

- Core remains a **linear** per-hypothesis pipeline (simple to reason about).
- Optional wrappers (preconditioner, staged evaluator) and a batch runner enable “cheap→expensive” staged pruning and informed sampling without turning the whole system into a general DAG.
- Termination and components remain pluggable; recipes stay thin convenience layers.
