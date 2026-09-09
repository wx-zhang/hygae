#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    data.train_files="$ROOT/data/navigation/train.parquet" \
    data.val_files="$ROOT/data/navigation/test.parquet" \
    data.max_trajectory_length=6000 \
    rollout_manager.max_turns=10 \
    trainer.experiment_name=navigation \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/navigation" \
    "$@"
