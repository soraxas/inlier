# Python RoMa examples

These scripts use the Python bindings in `python/inlier` together with RoMa
to produce real-world correspondences from the example image pair in `data/`.

## Setup

From the repo root:

```bash
uv pip install -e .
uv pip install romatch
```

Optional (fused local correlation kernel):

```bash
uv pip install romatch[fused-local-corr]
```

## Examples

Homography:

```bash
python py_examples/roma_homography.py
```

Fundamental:

```bash
python py_examples/roma_fundamental.py
```

Essential (requires intrinsics):

```bash
python py_examples/roma_essential.py --k1 path/to/K1.txt --k2 path/to/K2.txt
```

Notes:
- Essential matrix estimation expects normalized image coordinates.
- If you do not have calibration, use fundamental matrix instead.
