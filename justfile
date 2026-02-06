
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
