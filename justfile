
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

# Fetch all test data managed by inlier-data / pooch (downloads once, cached in OS cache dir)
fetch-data:
  uv run python -c "from inlier_data import TEST_DATA; [TEST_DATA.fetch(f) for f in TEST_DATA.registry]; print('All data fetched.')"

# Run all examples (skips those requiring file args or missing assets)
run-examples:
  #!/usr/bin/env bash
  cargo build --examples 2>&1 | grep "^error" || true
  pass=0; fail=0
  for ex in target/debug/examples/*; do
    [[ -x "$ex" && ! "$ex" == *-* ]] || continue
    name=$(basename "$ex")
    if output=$("$ex" 2>&1); then
      echo "PASS $name"; ((pass++))
    else
      echo "FAIL $name"; echo "  $(echo "$output" | tail -1)"; ((fail++))
    fi
  done
  echo ""
  echo "Results: $pass passed, $fail failed"
