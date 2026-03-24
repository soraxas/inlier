# inlier-library Specification

## Purpose
This specification defines the canonical public contract for the `inlier` workspace as a robust estimation library. It documents the current high-level estimation APIs, shared configuration and data abstractions, extensibility hooks, and feature-gated integrations so future changes can be evaluated against a stable baseline.

## Requirements

### Requirement: High-level estimation entry points
The library MUST expose high-level estimation entry points for homography, fundamental matrix, essential matrix, absolute pose, line, and rigid transform fitting. Each entry point MUST accept typed input data, an inlier threshold, and optional `MetasacSettings`, and MUST return either an `EstimationResult` containing the model, inliers, score, and iteration count or a descriptive error.

#### Scenario: Estimation succeeds with valid inputs
- **WHEN** a caller provides well-formed inputs for one of the supported estimation problems
- **THEN** the library returns an `EstimationResult` for the requested model family
- **AND** the result includes the fitted model, inlier indices, score, and iteration count

#### Scenario: Estimation rejects incompatible inputs
- **WHEN** a caller provides mismatched point counts or invalid dimensionality for a high-level estimation entry point
- **THEN** the library returns an error instead of producing a model
- **AND** the error describes the violated input expectation

### Requirement: Configurable MetaSAC behavior
The library MUST provide a `MetasacSettings` configuration object with defaults for robust estimation. Callers MUST be able to customize iteration bounds, confidence, inlier threshold, scoring, sampling, neighborhood reasoning, local optimization, final optimization, termination, inlier selection, and sampling attempts through this configuration surface.

#### Scenario: Callers rely on defaults
- **WHEN** a caller omits explicit settings and uses the default configuration
- **THEN** the library executes estimation with the default `MetasacSettings` values
- **AND** the defaults provide a complete runnable configuration without requiring additional setup

#### Scenario: Callers override algorithm choices
- **WHEN** a caller provides custom `MetasacSettings`
- **THEN** the library uses the supplied configuration values for the estimation pipeline
- **AND** optional point priors participate in scoring only when they align with the provided inputs

### Requirement: Shared geometric data abstractions
The library MUST expose shared data abstractions for geometric inputs, including `DataMatrix`, `Point2`, and `Point3`. `DataMatrix` MUST support construction from row-major input and MUST provide accessors for point count, dimensionality, and per-point values without requiring callers to depend on its internal storage layout.

#### Scenario: Callers construct data from row-major slices
- **WHEN** a caller creates a `DataMatrix` from row-major input data
- **THEN** the library accepts the provided shape and values
- **AND** subsequent estimation APIs can consume that matrix as input

#### Scenario: Callers inspect stored geometry
- **WHEN** a caller queries a `DataMatrix` for point counts, dimensions, or element values
- **THEN** the library exposes those values through `DataMatrix` methods
- **AND** callers do not need direct knowledge of the internal matrix representation

### Requirement: Extensible estimation architecture
The library MUST expose the core extension points needed to integrate custom robust-estimation behavior. Consumers MUST be able to implement or compose estimators, samplers, scoring strategies, local optimizers, termination criteria, inlier selectors, preconditioners, and pipeline types through the public API surface.

#### Scenario: Consumers implement custom components
- **WHEN** a consumer defines a custom estimator, sampler, scoring strategy, or optimizer against the published traits
- **THEN** the library provides the trait definitions needed to compile and integrate that component
- **AND** the component can participate in the broader estimation pipeline

#### Scenario: Consumers build on exported pipeline types
- **WHEN** a consumer uses the exported pipeline and extension types from the crate root
- **THEN** the library exposes those building blocks as part of the public API
- **AND** consumers can assemble workflows beyond the canned high-level entry points

### Requirement: Feature-gated integrations
The library MUST support optional integrations through Cargo features, including parallel execution, Python bindings, graph-cut optimization support, P3P-based absolute pose solving, Kornia-based pose solving, and point-cloud I/O support. These integrations MUST remain feature-gated so the base library can be used without enabling every optional dependency.

#### Scenario: Base library is used without optional integrations
- **WHEN** a consumer depends on the crate without enabling optional integration features
- **THEN** the core Rust estimation library remains usable
- **AND** feature-specific modules and dependencies are not required for the base workflow

#### Scenario: Consumers enable optional integrations
- **WHEN** a consumer enables one or more supported Cargo features
- **THEN** the corresponding integration surface becomes available to that build
- **AND** the enabled feature participates without changing the canonical behavior of unrelated APIs
