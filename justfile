
@default:
  just --list

gallery-html := "crates/inlier-gallery/index.html"

# Run the building floor-split pipeline on a point cloud (.ply).
# With inspect=1, dump 6-face visualisations of each stage
# (01_original, 02_aligned, 03_storeys, 04_walls) into <outdir> and open them.
#   just building SALON.ply
#   just building combined_pcd_both_floors_ds10.ply out 1
building input outdir="building-out" inspect="0":
  #!/usr/bin/env bash
  set -euo pipefail
  mkdir -p "{{outdir}}"
  cargo run --release -q -p spatialrust-inlier --example inspect_building \
    --features io,segmentation -- "{{input}}" "{{outdir}}"
  if [ "{{inspect}}" != "0" ]; then
    for s in 01_original 02_aligned 03_storeys 04_walls; do
      uv run --no-project --with matplotlib --with numpy \
        python scripts/render6.py "{{outdir}}/$s.vg" "{{outdir}}/$s.png" "$s"
    done
    echo "inspect images: {{outdir}}/*.png"
    [ "$(uname)" = "Darwin" ] && open "{{outdir}}"/*.png || true
  fi

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

gallery-html-threads := "crates/inlier-gallery/index-threads.html"

# EXPERIMENTAL multicore browser build (option #4). Compiles the whole gallery
# with wasm atomics on a nightly build-std toolchain and enables rayon + the
# wasm-bindgen-rayon worker pool. All the special flags live HERE as env vars,
# scoped to this invocation — never in .cargo/config.toml — so plain `wasm-dev`
# stays a stable, scalar build. Requires: `rustup toolchain install nightly`
# and `rustup component add rust-src --toolchain nightly`.
#
# Compiles and is configured (COOP/COEP via Trunk.toml, initThreadPool via
# threads-init.mjs). Browser runtime is UNVERIFIED: bevy-on-wasm-threads is
# upstream-unstable, so confirm rendering/segmentation actually work in a
# cross-origin-isolated browser before relying on it.
wasm-dev-threads:
  RUSTUP_TOOLCHAIN=nightly \
  CARGO_UNSTABLE_BUILD_STD="std,panic_abort" \
  RUSTFLAGS='--cfg getrandom_backend="wasm_js" -C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--export=__heap_base' \
    trunk serve {{gallery-html-threads}} --open

maturin-dev profile='--release' +args='':
  uv run maturin develop {{profile}} -F python {{args}}

test: test-rust test-rust-doc check-rust-python test-python

ensure-nextest:
  #!/usr/bin/env bash
  set -euo pipefail
  if cargo nextest --version >/dev/null 2>&1; then
    exit 0
  fi
  if cargo binstall --version >/dev/null 2>&1; then
    cargo binstall cargo-nextest --secure
  else
    cargo install cargo-nextest --locked
  fi

ensure-llvm-cov:
  #!/usr/bin/env bash
  set -euo pipefail
  if cargo llvm-cov --version >/dev/null 2>&1; then
    exit 0
  fi
  cargo install cargo-llvm-cov --locked

ensure-cargo-fuzz:
  #!/usr/bin/env bash
  set -euo pipefail
  if cargo fuzz --version >/dev/null 2>&1; then
    exit 0
  fi
  cargo install cargo-fuzz --locked

ensure-cargo-mutants:
  #!/usr/bin/env bash
  set -euo pipefail
  if cargo mutants --version >/dev/null 2>&1; then
    exit 0
  fi
  cargo install cargo-mutants --locked

test-rust profile='--release': ensure-nextest
  cargo nextest run --workspace --all-targets {{profile}}

check-rust-python:
  cargo check -p inlier --features python --lib

test-rust-doc:
  cargo test --doc --workspace

test-python profile='--release':
  uv run --no-project --with maturin maturin develop {{profile}} -F python --skip-install
  PYTHONPATH=python uv run --no-project --with pytest --with numpy --with-editable ../inlier-data pytest tests/python

coverage: ensure-nextest ensure-llvm-cov
  rustup component add llvm-tools-preview
  cargo llvm-cov nextest --workspace --all-targets

coverage-doctests lcov='lcov.info': ensure-nextest ensure-llvm-cov
  #!/usr/bin/env bash
  set -euo pipefail
  rustup toolchain install nightly --component llvm-tools-preview
  RUSTUP_TOOLCHAIN=nightly cargo llvm-cov --no-report nextest --workspace --all-targets
  RUSTUP_TOOLCHAIN=nightly cargo llvm-cov --no-report --doc --workspace
  RUSTUP_TOOLCHAIN=nightly cargo llvm-cov report --doctests --lcov --output-path "{{lcov}}"

# Coverage-guided public API input safety fuzzing. Use a bounded run locally
# or in scheduled CI; corpus and crash artifacts are intentionally ignored.
fuzz-public-api duration='60': ensure-cargo-fuzz
  rustup toolchain install nightly --profile minimal
  cargo +nightly fuzz run public_api -- -max_total_time={{duration}}

# Target control flow and validation rather than expensive floating-point kernels.
mutants path='src/core.rs': ensure-cargo-mutants
  cargo mutants --file "{{path}}"

# Run deterministic, synthetic end-to-end estimator benchmarks. Real-world benchmark assets
# belong in the inlier-data submodule.
bench-estimators:
  cargo bench --bench estimators

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
