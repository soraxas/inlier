# Python bindings

The Python bindings live in [python/inlier](../../src/python/mod.rs) and expose the same
pipeline as the Rust API via PyO3. The bindings mirror `SuperRansac` components through adapter
classes like `EstimatorAdapter`, `SamplerAdapter`, and `ScoringAdapter` and support runtime
enums defined in [src/choices.rs](../../src/choices.rs).

Key exposed helpers:

- `pysuperansac.estimate_homography`
- `pysuperansac.estimate_fundamental_matrix`
- `pysuperansac.estimate_essential_matrix`
- `pysuperansac.estimate_rigid_transform`
- `pysuperansac.estimate_absolute_pose`

The bindings wrap the Rust settings structures and allow you to pass `point_priors`, select
samplers/scorers via the `ScoringType`/`SamplerType` enums, and hook custom Python callbacks.

```python
import inlier as pysuperansac

settings = pysuperansac.RansacSettings()
settings.max_iterations = 2000
model = pysuperansac.estimate_homography(correspondences, (800, 600), settings)
```

The crate also ships a [tests/python/test_bindings.py](../../tests/python/test_bindings.py) suite to ensure feature parity with the
[superansac_c++/](../../superansac_c++/) C++ references. Embed part of the binding file to keep docs fresh:
