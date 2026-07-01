
@default:
  just --list

gallery-html := "crates/inlier-gallery/index.html"

# One-time: install the wasm target and trunk (browser build toolchain).
# Run this once before `wasm-dev` / `wasm-build`.
wasm-setup:
  rustup target add wasm32-unknown-unknown
  cargo binstall -y trunk || cargo install --locked trunk

# Serve the inlier-gallery as a browser app with hot reload (http://localhost:8080).
# Debug profile: trunk skips wasm-opt, so rebuilds stay fast. Run `just wasm-setup` first.
wasm-dev:
  trunk serve {{gallery-html}} --open

# Build an optimized wasm bundle into dist/. NOTE: --release runs wasm-opt over the
# full bevy wasm module, which is slow and memory-hungry (untested end-to-end here).
wasm-build:
  trunk build --release {{gallery-html}}

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
