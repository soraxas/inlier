# Pipeline components

The pipeline exposed by `SuperRansac` is entirely driven by [RansacSettings](../../src/settings.rs)
and its enum members. Each stage maps to a concrete implementation defined in the Rust modules
listed below.

**Table of contents**
- [Sampling](#sampling)
- [Neighborhood graphs](#neighborhood-graphs)
- [Estimators and models](#estimators-and-models)
- [Scoring](#scoring)
- [Configuration enums](#configuration-enums)
- [Quick overview: staged pipeline](#quick-overview-staged-pipeline)
- [Pipeline diagram](#pipeline-diagram)
- [Detailed notes and equations](#detailed-notes-and-equations)
  - [Preprocessing](#preprocessing)
  - [Sampling](#sampling-1)
  - [Sample degeneracy checks](#sample-degeneracy-checks)
  - [Minimal model estimation](#minimal-model-estimation)
  - [Model degeneracy checks](#model-degeneracy-checks)
  - [Scoring and residual objectives](#scoring-and-residual-objectives)
  - [Preemptive verification](#preemptive-verification)
  - [Optimization (LO/FO)](#optimization-lofo)
  - [Non-minimal estimation](#non-minimal-estimation)
  - [Termination](#termination)

## Sampling

- **Uniform**: `UniformRandomSampler` randomly draws minimal samples. See
  [src/samplers/uniform.rs](../../src/samplers/uniform.rs).
- **PROSAC**: `ProsacSampler` uses inlier probability ordering to accelerate good samples.
  Implementation lives in [src/samplers/prosac.rs](../../src/samplers/prosac.rs).
- **NAPSAC / Progressive NAPSAC**: Spatial samplers that choose neighborhoods from
  [NeighborhoodGraph](../../src/samplers/neighborhood.rs) helpers (`Grid` or `Usearch`).
- **Importance / AR samplers**: Use adaptive reordering or custom importance weights.
  The logic spans [src/samplers/importance.rs](../../src/samplers/importance.rs) and
  [src/samplers/adaptive_reordering.rs](../../src/samplers/adaptive_reordering.rs).

Selecting a sampler is done via `SamplerType` in `RansacSettings`: each variant maps to the
matching sampler implementation. Pass the `settings.sampler` enum to core or to the Python
bindings so runtime configuration is easy.

## Neighborhood graphs

Spatial samplers depend on neighborhood graphs. The crate provides a `GridNeighborhoodGraph`
and a `UsearchNeighborhoodGraph` that uses the `usearch` crate for approximate nearest
neighbors. Both implement the same [NeighborhoodGraph](../../src/samplers/neighborhood.rs)
trait, so you can swap or extend the graph implementation.

Neighborhood graphs back samplers such as `NapsacSampler` and `ProgressiveNapsacSampler` and
are exposed through the `NeighborhoodType` enum in `RansacSettings`. The `Grid` variant
connects to a lightweight uniform grid, while the `BruteForce` and `Flann*` variants rely on
explicit neighbor lists (the latter currently use the `usearch` index as a stand-in for FLANN).

## Estimators and models

Built-in estimators live under [src/estimators](../../src/estimators/mod.rs) and cover:

1. **Homography** – 4/8-point estimation for image-to-image transforms.
2. **Fundamental matrix** – 7-/8-point solvers plus bundle-adjustment-like refinements.
3. **Essential matrix** – Nistér five-point solver with optional bundle adjustment tweaks.
4. **Absolute pose** – Lambda Twist (`p3p` feature) or DLT fallback plus `kornia-pnp` refinements.
5. **Rigid transform** – 3D-to-3D Procrustes alignment.
6. **Line** – simple line models for quick experiments.

`Estimator` implementations can return multiple solutions per sample (e.g., essential matrix) and
provide both minimal and non-minimal (`estimate_model_nonminimal`) interfaces so local
optimizers can reuse inliers.

## Scoring

Scores are produced by the `Scoring` trait; built-in implementations in
[src/scoring.rs](../../src/scoring.rs) include:

- `RansacInlierCountScoring` – the classic inlier counter.
- `MsacScoring` – truncated squared residual cost negated so higher is always better.
- `MinpranScoring` – per-point weighted inlier counting for MinPRAN-style branches.

All scorers store the same `Score` struct (inlier count + value) and can attach per-point
priors via their `with_priors` helpers. The sigma-consensus++ infrastructure builds
precomputed `sigma_lut` tables during `build.rs` to evaluate gamma functions cheaply
(see the Algorithms chapter for details).

## Configuration enums

`RansacSettings` centralizes control of the pipeline:

- `scoring`: choose `ScoringType` (`Magsac`, `Minpran`, `Acransac`, etc.).
- `sampler`: choose `SamplerType`.
- `neighborhood`: choose `NeighborhoodType` for spatial samplers.
- `local_optimization` / `final_optimization`: select `LocalOptimizationType` (more in the next
  chapter).
- `termination_criterion`: currently `TerminationType::Ransac`.
- `inlier_selector`: pick an inlier selector strategy, e.g., `SpacePartitioningRansac`.
  - `point_priors`: optional per-point weights that many scoring functions consume.

See [src/settings.rs](../../src/settings.rs) for the full definition and defaults that match
the C++ implementation.

## Quick overview: staged pipeline

Text-only high-level view for **H/F/E**, absolute pose, and rigid transform (minimal → score → refine loop):

1) Preprocess: normalize coordinates per problem; scale τ consistently.
2) Sample: quality-ranked progressive sampling; spatially local for H/PnP/3D–3D; avoid local for F/E.
3) Sample degeneracy: reject twisted/collinear quads (H); collinear 3D triplets (PnP/3D–3D); usually skip for F/E.
4) Minimal solver: H 4-pt DLT; F 7-pt; E 5-pt; PnP P3P; 3D–3D Procrustes (3 pts).
5) Model degeneracy: det(H) sanity; det(R)≈+1 for rigid/PnP; nothing extra for F/E.
6) Score: residual-based scoring (transfer/Sampson/reprojection/Euclidean) with MSAC/MAGSAC/MinPRAN.
7) Preemptive bail-out: stop scoring if optimistic bound ≤ best.
8) Optimize: LO (graph-based or nested resampling); FO IRLS with threshold annealing.
9) Non-minimal fits: DLT (H); 8-pt + LM Sampson (F); pose + LM (E); EPnP + LM (PnP); Procrustes (3D–3D).
10) Termination: RANSAC confidence rule.

## Pipeline diagram

```mermaid
flowchart TD
    In[Inputs: correspondences (+K, +τ)]
    In --> P1
    subgraph MainLoop[Model search loop]
      P1[1) Preprocess\n- normalize coords\n- scale τ accordingly]
      S2[2) Sampling\n- guided (ranked)\n- maybe spatial-local\n  (H / PnP / 3D-3D)\n- avoid local for F/E]
      D3[3) Sample degeneracy\n- H: twisted/collinear\n- PnP/3D-3D: collinear\n- F/E: usually skip]
      M4[4) Minimal solver\n- H: 4pt DLT\n- F: 7pt\n- E: 5pt\n- PnP: P3P\n- 3D-3D: 3pt SVD]
      G5[5) Model degeneracy\n- det(H) range\n- det(R) ≈ +1]
      S6[6) Score model on all points]
      B7[7) Preemptive bail-out]
      Q{Better than best?}
      LO8[8) Local Opt (LO)\n- graph / nested]
      NM9[9) Nonminimal + refine]
      T10[10) Terminate?\nconfidence rule]

      P1 --> S2 --> D3 --> M4 --> G5 --> S6 --> B7 --> Q
      Q -->|no| T10
      Q -->|yes| LO8 --> NM9 --> Q
      T10 -->|continue| S2
    end
    T10 -->|stop| End([Done])
```

## Detailed notes and equations

### Preprocessing
Goal: keep solvers stable and thresholds meaningful.
- **2D–2D unknown intrinsics (H, F)**: Hartley normalization (centroid → origin, mean distance → √2).
- **Known intrinsics (E, PnP)**: pixels → normalized camera coordinates via \(K\) (focal length + principal point).
- **3D–3D**: translate points to centroid; do not scale (preserve scene scale).
- **Threshold scaling**: apply the same scale used in normalization so τ matches the geometric error in the original space.

### Sampling
Goal: draw all-inlier minimal samples quickly.
- Quality-ranked progressive sampling; optional spatial coherence via overlapping uniform grids, confined to the grid cell of the first point.
- Problem-specific: spatial locality helps H/PnP/3D–3D; avoid locality for F/E to reduce degeneracy.
- Minimal sizes: H=4, F=7 (or 8 nonminimal init), E=5, PnP=3 (P3P), 3D–3D=3.

### Sample degeneracy checks
Goal: reject bad minimal sets pre-solver.
- **H**: reject twisted/self-intersecting or collinear quads. A common check: with a consistent cyclic ordering, ensure non-adjacent edges do not intersect (orientation / segment-intersection tests using 2D cross products: `orient(C,D,A)` and `orient(C,D,B)` have opposite signs, and the symmetric check for `A,B` against `C,D`).
- **Rigid 3D–3D / PnP**: reject collinear 3D triplets (‖(P2−P1)×(P3−P1)‖ ≈ 0); for 3D–3D apply on both corresponding triplets.
- **F / E**: typically no pre-solver test.

### Minimal model estimation
Goal: compute a hypothesis per minimal sample.
- **H**: 4-pt DLT (normalized) → 8 equations; fix scale (e.g., set H33=1) and solve an 8×8 linear system.
- **F**: 7-pt solver → build 7×9 constraint matrix, take 2D null space (F1,F2), enforce det(αF1+(1−α)F2)=0 cubic (1–3 solutions).
- **E**: 5-pt solver → 4D null space, Demazure constraints → 10th-degree polynomial (multiple solutions).
- **Rigid 3D–3D**: Procrustes (SVD).
- **Absolute pose (P3P)**: P3P solver (multiple pose candidates).

### Model degeneracy checks
Goal: discard invalid solved models early.
- **H**: |det(H)| within plausible range (reject collapsing or unstable homographies).
- **Rigid / PnP**: det(R) ≈ +1 (reject reflections).
- **F / E**: rely on solver constraints or downstream scoring.

### Scoring and residual objectives
Goal: robustly rank hypotheses against noise/outliers.
- Residual definitions:
  - **Homography transfer error (symmetric)**:
    \[
    r^2 = \left\lVert \mathbf{x}' - \pi(H \mathbf{x}) \right\rVert^2 + \left\lVert \mathbf{x} - \pi(H^{-1} \mathbf{x}') \right\rVert^2,
    \]
    with \(\pi\) dividing by homogeneous scale.
  - **F/E Sampson error** (normalized camera coordinates for \(E\)):
    \[
    r^2 = \frac{(\mathbf{x}'^{\top} F \mathbf{x})^2}{(F\mathbf{x})_1^2 + (F\mathbf{x})_2^2 + (F^{\top}\mathbf{x}')_1^2 + (F^{\top}\mathbf{x}')_2^2}.
    \]
  - **PnP reprojection error**:
    \[
    r^2 = \left\lVert \mathbf{u} - \pi\!\left(K [R \,|\, \mathbf{t}] \mathbf{X}\right) \right\rVert^2.
    \]
  - **Rigid 3D–3D error**:
    \[
    r^2 = \left\lVert R \mathbf{X} + \mathbf{t} - \mathbf{Y} \right\rVert^2.
    \]
- Scoring families: inlier count, truncated costs (MSAC), noise-scale–robust (MAGSAC-style). Define per-point contribution \(q(r)\) with cap \(q_{\max}\); preemptive verification uses \(q_{\max}\) as the optimistic bound.

### Preemptive verification
Goal: stop scoring hopeless models early.
- After k points: \(s_{\text{optimistic}} = s(k) + (n-k)\,q_{\max}\); if \(s_{\text{optimistic}} \le s_{\text{best}}\), stop scoring.
- Homography can also use space-partitioning verifiers for extra speed.

### Optimization (LO/FO)
Goal: refine promising models.
- **Local optimization (in-loop)**:
  - Moderate N: spatial-coherence LO (4D grid neighbors on (x,y,x’,y’)), graph-cut–style inlier selection, then refit.
  - Large N: nested resampling LO (e.g., sample ~7·m points from current inliers; m = minimal size).
- **Final optimization**: IRLS with robust weights (e.g., Cauchy) and threshold annealing; halve the threshold each IRLS iteration to tighten consensus.

### Non-minimal estimation
Goal: higher-accuracy fits from larger inlier sets.
- **H**: normalized nonminimal DLT.
- **F**: 8-pt init + LM on Sampson with rank-2 parameterization.
- **E**: algebraic init → pose disambiguation (chirality) → LM on 5-DoF pose (Sampson).
- **PnP**: EPnP init → LM on reprojection.
- **3D–3D**: Procrustes on all inliers.

### Termination
Goal: stop when further improvement is unlikely.
- Standard RANSAC confidence rule using current inlier ratio and desired confidence (e.g., 99.9%).

## Batch/staged “DAG-in-a-loop” (experimental architecture)

If you want BIT*/AIT*-style batching with cheap→expensive staged verification and informed sampling without turning the whole loop into a DAG, run an **epoch DAG** inside the outer RANSAC loop and carry state across epochs:

- **Carry state across epochs**: best model/score, per-point priors `p_i` (inlier likelihood / quality rank), staged subsets (P1/P2/P3/all), stopping bound.
- **Epoch DAG (acyclic)**:
  - Precompute/normalize per epoch (coords, τ scaling, grids, subsets).
  - Batch sample minimal sets with a mixture policy (priors + quality rank + uniform exploration); cheap sample degeneracy checks.
  - Minimal solve (expand multi-solution solvers) → cheap model sanity checks.
  - **Cheap eval + optimistic bound** on a small subset (P1), push into a priority queue keyed by bound.
  - Best-first staged verification on larger subsets (P2 → P3 → all), pruning any hypothesis whose upper bound cannot beat the incumbent.
  - Run LO/nonminimal refinement only on the top few survivors; update best model/score.
  - Update priors `p_i` from final residuals (boost consistent inliers, down-weight strong outliers); optionally smooth spatially for H/PnP/3D–3D.
- **Why it helps**: better incumbents tighten bounds; most hypotheses die after P1/P2; priors improve sampling over epochs.
- **Tunable defaults**: batch size 64–512; staged subsets ~32 → 128 → 512 → all; keep top ~10–20% per stage; LO on top 1–3; 10–30% exploratory uniform samples to avoid lock-in.
- **Execution**: still a linear pipeline per hypothesis; staged evaluation + best-first ordering + updated priors give DAG-like pruning without complicating the core API.
