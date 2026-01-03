
@default:
  just --help

test +args='':
  cargo test {{args}} --features python,examples
