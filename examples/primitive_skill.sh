#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    data.train_files="$ROOT/data/primitive_skill/train.parquet" \
    data.val_files="$ROOT/data/primitive_skill/test.parquet" \
    data.max_trajectory_length=6000 \
    rollout_manager.max_turns=10 \
    trainer.experiment_name=primitive_skill \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/primitive_skill" \
    "$@"
