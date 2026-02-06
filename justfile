
@default:
  just --list

maturin-dev profile='--release':
  uv run maturin develop {{profile}} --locked --all-features

[parallel]
test: test-rust test-python

test-rust profile='--release':
  cargo nextest run --all-features {{profile}}
test-python:
  uv run pytest
