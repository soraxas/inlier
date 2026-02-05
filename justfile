
@default:
  just --help


[parallel]
test: test-rust test-python

test-rust profile='--release':
  cargo nextest run --all-features {{profile}}
test-python:
  uv run pytest
