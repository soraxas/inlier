
@default:
  just --list

maturin-dev profile='--release' +args='':
  uv run maturin develop {{profile}} -F python {{args}}

[parallel]
test: test-rust test-python

test-rust profile='--release':
  cargo nextest run {{profile}}
test-python:
  uv run pytest

example-assets:
  cargo run --example homography_estimation
  uv run --group dev python python/demo_homography_scene.py
  cargo run --example rigid_transform
  uv run --group dev python python/demo_rigid_transform_scene.py
  cargo run --example plot_line_fitting --features examples
