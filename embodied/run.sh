#!/usr/bin/env bash
# One entry point for the embodied track.
#
#   ./embodied/run.sh setup    install dependencies and the simulator
#   ./embodied/run.sh data     fetch LIBERO-Long demonstrations
#   ./embodied/run.sh sft      behaviour cloning on the official demonstrations
#   ./embodied/run.sh train    GRPO with the same two corrections
#   ./embodied/run.sh eval     success rate on seen and unseen initial states
#
# Unlike the AutoResearch track this runs on a single node: the simulator is
# cheap enough that fencing it onto its own machine buys nothing.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${WMRL_CONFIG:-embodied/configs/libero_long.yaml}"
DATA_ROOT="${WMRL_DATA:-$REPO/data/libero}"
CKPT_ROOT="${WMRL_CKPT:-$REPO/checkpoints/embodied}"
PY="${WMRL_PYTHON:-python3}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

cmd_setup() {
  say "Installing dependencies"
  $PY -m pip install -r requirements.txt
  PYTHONPATH="$REPO" $PY -m pytest tests -q
  cat <<EOF

The simulator and the policy backbone are not pinned here:

  LIBERO      the benchmark and its demonstrations
  MiniVLA     the 1B vision-language-action policy
  Robometer   the off-the-shelf progress predictor used as the world model

See embodied/README.md. Headless rendering needs a working EGL or OSMesa stack,
which is the step that usually fails first on a fresh machine.
EOF
}

cmd_data() {
  say "Fetching LIBERO-Long demonstrations into $DATA_ROOT"
  mkdir -p "$DATA_ROOT"
  PYTHONPATH="$REPO" $PY -m embodied.data --out "$DATA_ROOT" "$@"
}

cmd_sft() {
  say "Behaviour cloning"
  PYTHONPATH="$REPO" $PY embodied/sft.py \
    --data "$DATA_ROOT" --out "$CKPT_ROOT/sft" "$@"
}

cmd_train() {
  [ -d "$CKPT_ROOT/sft" ] || die "no SFT checkpoint in $CKPT_ROOT/sft; run: run.sh sft
Every RL run starts from the same SFT initialisation, so the spread between
reward signals is attributable to the signal rather than to the starting point."
  say "GRPO with online debiasing and inverse-variance denoising"
  PYTHONPATH="$REPO" $PY embodied/train_grpo.py \
    --config "$CONFIG" --init "$CKPT_ROOT/sft" --out "$CKPT_ROOT/wmrl" "$@"
}

cmd_eval() {
  local ckpt="${1:-$CKPT_ROOT/wmrl}"
  shift || true
  say "Evaluating $ckpt"
  PYTHONPATH="$REPO" $PY -m embodied.evaluate \
    --checkpoint "$ckpt" --data "$DATA_ROOT" "$@"
}

usage() {
  cat <<EOF
WMRL, embodied track. LIBERO-Long, MiniVLA-1B.

  setup                install dependencies
  data                 fetch demonstrations
  sft                  behaviour cloning, the shared starting point
  train                GRPO with both corrections
  eval [checkpoint]    success rate, in-domain and out-of-distribution

Configuration
  WMRL_CONFIG  $CONFIG
  WMRL_DATA    $DATA_ROOT
  WMRL_CKPT    $CKPT_ROOT

Single node. The reward is a dense per-frame progress prediction; the anchor is
the one sparse success the simulator returns at the end of a rollout.
EOF
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  data)  shift; cmd_data  "$@" ;;
  sft)   shift; cmd_sft   "$@" ;;
  train) shift; cmd_train "$@" ;;
  eval)  shift; cmd_eval  "$@" ;;
  ""|-h|--help|help) usage ;;
  *) die "unknown command: $1 (try: run.sh help)" ;;
esac
