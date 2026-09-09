#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    data.train_files="$ROOT/data/frozenlake/train.parquet" \
    data.val_files="$ROOT/data/frozenlake/test.parquet" \
    data.max_trajectory_length=2400 \
    rollout_manager.max_turns=3 \
    trainer.experiment_name=frozenlake \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/frozenlake" \
    "$@"
