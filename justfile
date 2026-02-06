
@default:
  just --list

maturin-dev:
  uv run maturin develop --release --locked --all-features

[parallel]
test: test-rust test-python

test-rust profile='--release':
  cargo nextest run --all-features {{profile}}
test-python:
  uv run pytest
