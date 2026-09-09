#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    trainer.experiment_name=sokoban_rl4vlm \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/sokoban_rl4vlm" \
    algorithm.adv_estimator=rl4vlm \
    algorithm.gamma=0.99 \
    algorithm.lam=1.0 \
    algorithm.thought_prob_coef=0.5 \
    critic.use_reward_mask=true \
    rollout_manager.n_trajectory=1 \
    "$@"
