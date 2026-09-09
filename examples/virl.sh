#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    data.train_files="$ROOT/data/virl/train.parquet" \
    data.val_files="$ROOT/data/virl/test.parquet" \
    data.max_trajectory_length=12000 \
    rollout_manager.max_turns=10 \
    trainer.experiment_name=virl \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/virl" \
    "$@"
