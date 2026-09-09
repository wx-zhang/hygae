#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/train.sh" \
    trainer.experiment_name=sokoban_grpo \
    trainer.default_local_dir="$ROOT/checkpoints/hygae/sokoban_grpo" \
    algorithm.adv_estimator=masked_grpo \
    algorithm.gamma=1.0 \
    algorithm.thought_prob_coef=1.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.rollout.n=1 \
    rollout_manager.n_trajectory=8 \
    "$@"
