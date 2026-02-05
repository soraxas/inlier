
@default:
  just --help

test:
  cargo nextest run --all-features
