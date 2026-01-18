
@default:
  just --list

test +args='':
  cargo test {{args}} --all-features

coverage:
  cargo tarpaulin --no-dead-code --engine llvm
