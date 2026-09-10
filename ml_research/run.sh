#!/usr/bin/env bash
# One entry point for the AutoResearch track. Run it with no arguments to see
# what each step does and whether your environment is ready for it.
#
#   ./ml_research/run.sh setup     install dependencies
#   ./ml_research/run.sh data      download and prepare MLE-Dojo and DSBench
#   ./ml_research/run.sh train     start this node's part of a training run
#   ./ml_research/run.sh eval      score a checkpoint on the held-out sets
#
# `train` is launched identically on every node. Each works out its own role
# from WMRL_HOSTS, so there is no separate command for the trainer.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${WMRL_CONFIG:-ml_research/configs/wmrl_9b.yaml}"
DATA_ROOT="${WMRL_DATA:-$REPO/data}"
MODEL_ROOT="${WMRL_MODELS:-$REPO/models}"
PY="${WMRL_PYTHON:-python3}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

need_env() {
  local missing=()
  for v in "$@"; do [ -n "${!v:-}" ] || missing+=("$v"); done
  if [ ${#missing[@]} -gt 0 ]; then
    die "unset: ${missing[*]}
  WMRL_HOSTS   comma-separated hostnames in role order: trainer,world_model,sandbox
  WMRL_STORE   a directory every node mounts, or s3://bucket/prefix  # audit-allow: documented example
  WMRL_RUN_ID  a name unique to this run

Example:
  export WMRL_HOSTS=node0,node1,node2
  export WMRL_STORE=/mnt/shared/wmrl
  export WMRL_RUN_ID=wmrl-9b-\$(date +%Y%m%d)"
  fi
}

# --------------------------------------------------------------------- steps

cmd_setup() {
  say "Installing dependencies"
  $PY -m pip install -r requirements.txt
  say "Checking the algorithm library"
  PYTHONPATH="$REPO" $PY -m pytest tests -q
  cat <<EOF

Dependencies installed and the algorithm tests pass.

The training loop additionally needs an inference backend and an RL trainer,
which are large and version-sensitive enough that they are not pinned here.
See ml_research/README.md for the versions these runs were built against.
EOF
}

cmd_data() {
  say "Preparing data under $DATA_ROOT"
  mkdir -p "$DATA_ROOT"
  [ -n "${KAGGLE_USERNAME:-}${KAGGLE_KEY:-}" ] || warn \
    "no Kaggle credentials in the environment; competition downloads will fail"

  PYTHONPATH="$REPO" $PY -m ml_research.data.prepare \
    --tasks ml_research/configs/tasks_train.txt \
    --out "$DATA_ROOT" \
    "$@"

  say "Fetching model weights into $MODEL_ROOT"
  mkdir -p "$MODEL_ROOT"
  PYTHONPATH="$REPO" $PY -m ml_research.data.fetch_models --out "$MODEL_ROOT" "$@"
}

cmd_train() {
  need_env WMRL_HOSTS WMRL_STORE WMRL_RUN_ID
  say "Starting this node's role for run $WMRL_RUN_ID"
  export WMRL_DATA="$DATA_ROOT" WMRL_MODELS="$MODEL_ROOT"
  PYTHONPATH="$REPO" exec $PY -m ml_research.cluster.launch --config "$CONFIG" "$@"
}

cmd_plan() {
  need_env WMRL_HOSTS
  PYTHONPATH="$REPO" $PY -m ml_research.cluster.launch --config "$CONFIG" --dry-run
}

cmd_eval() {
  local ckpt="${1:-}"
  [ -n "$ckpt" ] || die "usage: run.sh eval <checkpoint-path> [--benchmark mle|dsbench|both]"
  shift
  say "Scoring $ckpt on the held-out sets"
  PYTHONPATH="$REPO" $PY -m ml_research.eval.eval_checkpoint \
    --checkpoint "$ckpt" --data "$DATA_ROOT" "$@"
}

usage() {
  cat <<EOF
WMRL, AutoResearch track.

  setup                     install dependencies and run the algorithm tests
  data                      download and prepare MLE-Dojo and DSBench
  plan                      resolve the topology and print this node's role
  train                     run this node's role in a training job
  eval <checkpoint>         score a checkpoint on the held-out sets

Configuration
  WMRL_CONFIG   $CONFIG
  WMRL_DATA     $DATA_ROOT
  WMRL_MODELS   $MODEL_ROOT
  WMRL_HOSTS    ${WMRL_HOSTS:-(unset)}
  WMRL_STORE    ${WMRL_STORE:-(unset)}
  WMRL_RUN_ID   ${WMRL_RUN_ID:-(unset)}

A run is three nodes. Export the three variables on each, then run
\`run.sh train\` on all three; roles follow the order of WMRL_HOSTS.
Start with \`run.sh plan\` to check each node agrees on who it is.
EOF
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  data)  shift; cmd_data  "$@" ;;
  plan)  shift; cmd_plan  "$@" ;;
  train) shift; cmd_train "$@" ;;
  eval)  shift; cmd_eval  "$@" ;;
  ""|-h|--help|help) usage ;;
  *) die "unknown command: $1 (try: run.sh help)" ;;
esac
