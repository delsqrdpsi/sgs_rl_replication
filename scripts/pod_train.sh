#!/usr/bin/env bash
# Train on a CUDA machine (e.g. a RunPod RTX 5090). Run from the repo root:
#   SAMPLER=sgs HOURS=8 bash scripts/pod_train.sh
# Checks the GPU path with a 2-iteration run first, then starts the real run in the background.
set -euo pipefail

SAMPLER=${SAMPLER:-sgs}  # sgs | uniform
ENVS=${ENVS:-10000}
ITERS=${ITERS:-100000}
HOURS=${HOURS:-8}
SEED=${SEED:-0}
PROJECT=${PROJECT:-sgs-rod}
export PYTHONPATH=src

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== GPU check: 2 iterations with $ENVS environments"
WANDB_MODE=disabled uv run --no-sync python scripts/train.py --sampler "$SAMPLER" --num-envs "$ENVS" \
  --iterations 2 --save-interval 1 --eval-size 1024 --run-dir runs/gpu-check
rm -rf runs/gpu-check

name="$SAMPLER-$ENVS-s$SEED-$(date +%Y%m%d-%H%M%S)"
mkdir -p runs
echo "== Training run $name (W&B offline; sync later with: uv run wandb sync runs/$name/wandb/offline-run-*)"
WANDB_MODE=offline nohup uv run --no-sync python scripts/train.py --sampler "$SAMPLER" --num-envs "$ENVS" \
  --iterations "$ITERS" --max-hours "$HOURS" --seed "$SEED" --run-dir "runs/$name" --wandb-project "$PROJECT" \
  > "runs/$name.log" 2>&1 &
echo "started: tail -f runs/$name.log"
