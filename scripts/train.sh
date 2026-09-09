#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# GPU and batch-size settings — edit this block for your hardware.
# Global batch sizes should usually scale with the total number of GPUs.
# If you run out of memory, reduce the per-GPU micro-batch sizes first.
# =============================================================================
NUM_GPUS_PER_NODE=4
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
MICRO_BATCH_SIZE_PER_GPU=2

# Make this checkout and its patched VERL visible to local and Ray workers.
export PYTHONPATH="$REPO_ROOT/third_party/verl:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=42
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_PROJECT=hygae
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/outputs/wandb}"

python -m hygae.trainer.main_ppo \
    data.train_files="$REPO_ROOT/data/sokoban/train.parquet" \
    data.val_files="$REPO_ROOT/data/sokoban/test.parquet" \
    data.seed=42 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    algorithm.adv_estimator=hygae \
    algorithm.gamma=0.99 \
    algorithm.lam=1.0 \
    algorithm.alpha=0.5 \
    critic.use_reward_mask=false \
    critic.model.value_head_init_std=0.002 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    critic.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    critic.forward_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    rollout_manager.n_trajectory=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NUM_GPUS_PER_NODE" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name=sokoban \
    trainer.default_local_dir="$REPO_ROOT/checkpoints/hygae/sokoban" \
    trainer.resume_mode=auto \
    trainer.total_training_steps=300 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.val_before_train=false \
    trainer.val_generations_to_log_to_wandb=8 \
    "$@"
