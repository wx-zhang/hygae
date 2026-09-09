#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    trainer.experiment_name=sokoban_ppo \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/sokoban_ppo" \
    algorithm.adv_estimator=masked_gae \
    algorithm.gamma=0.99 \
    algorithm.lam=1.0 \
    algorithm.thought_prob_coef=1.0 \
    critic.use_reward_mask=false \
    rollout_manager.n_trajectory=1 \
    "$@"
