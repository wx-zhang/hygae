# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from enum import Enum

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    MASKED_GAE = "masked_gae"
    BI_LEVEL_GAE = "bi_level_gae"
    TURN_WISE_GAE = "turn_wise_gae"
    GRPO = "grpo"
    MASKED_GRPO = "masked_grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    MULTI_TURN_GRPO = "multi_turn_grpo"
    HYGAE = "hygae"
    MASKED_REINFORCE = "masked_reinforce"
    RL4VLM = "rl4vlm"


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma=1.0,
    lam=1.0,
    high_level_gamma=1.0,
    alpha=0.0,
    aggregate_reward_in_turn=False,
    thought_prob_coef=1.0,
):
    """Compute advantages using the specified estimator.

    Two masks are used throughout:
      - attention_mask: 1 for all real (non-pad) tokens in the full sequence.
      - loss_mask: 1 for model-generated content tokens only (subset of attention_mask).

    Additionally, end_of_response_position_mask is a sparse turn-boundary marker
    used by multi-turn estimators.

    The actor's PPO loss uses data.batch["response_mask"]. We overwrite it with
    loss_mask here so the actor trains only on model-generated tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(-1)

    # ── Required masks ──────────────────────────────────────────────────
    attention_mask = data.batch["attention_mask"]
    attention_response_mask = attention_mask[:, -response_length:]

    # loss_mask: model-generated content tokens only (required)
    # Rollouts expose this as either loss_mask or response_mask.
    if "loss_mask" in data.batch:
        loss_mask = data.batch["loss_mask"][:, -response_length:]
    elif "response_mask" in data.batch:
        loss_mask = data.batch["response_mask"][:, -response_length:]
    else:
        raise KeyError("Neither 'loss_mask' nor 'response_mask' found in batch. "
                        "The rollout must produce a content-only mask.")

    # Overwrite response_mask so verl's actor PPO loss uses content-only mask
    data.batch["response_mask"] = loss_mask

    # ── Optional fields ─────────────────────────────────────────────────
    token_level_rewards = data.batch["token_level_rewards"]
    values = data.batch.get("values", None)

    # end_of_response_position_mask: sparse turn-boundary marker
    # Required by multi-turn estimators, optional otherwise
    reward_mask = data.batch.get("end_of_response_position_mask", None)
    if reward_mask is not None:
        reward_mask = reward_mask[:, -response_length:]

    uid = data.non_tensor_batch.get("uid")

    # ── Dispatch to estimator ───────────────────────────────────────────
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=attention_response_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == AdvantageEstimator.MASKED_GAE:
        advantages, returns = compute_gae_advantage_return_with_loss_mask(
            token_level_rewards=token_level_rewards,
            values=values,
            loss_mask=loss_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == AdvantageEstimator.BI_LEVEL_GAE:
        advantages, returns = compute_bi_level_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            loss_mask=loss_mask,
            gamma=gamma,
            lam=lam,
            high_level_gamma=high_level_gamma,
            reward_mask=reward_mask,
        )
    elif adv_estimator == AdvantageEstimator.HYGAE:
        advantages, returns = compute_hygae_return(
            token_level_rewards=token_level_rewards,
            values=values,
            loss_mask=loss_mask,
            gamma=gamma,
            lam=lam,
            reward_mask=reward_mask,
            alpha=alpha,
            aggregate_reward_in_turn=aggregate_reward_in_turn,
        )
    elif adv_estimator == AdvantageEstimator.TURN_WISE_GAE:
        advantages, returns = compute_turn_wise_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            loss_mask=loss_mask,
            reward_mask=reward_mask,
            lam=lam,
            high_level_gamma=high_level_gamma,
        )
    elif adv_estimator == AdvantageEstimator.MASKED_GRPO:
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=loss_mask,
            index=uid,
        )
    elif adv_estimator == AdvantageEstimator.MASKED_REINFORCE:
        advantages, returns = compute_masked_reinforce_advantage(
            token_level_rewards=token_level_rewards,
            loss_mask=loss_mask,
            gamma=gamma,
        )
    elif adv_estimator == AdvantageEstimator.RL4VLM:
        if values is None or reward_mask is None:
            raise ValueError("RL4VLM requires critic values and turn-boundary reward masks")
        advantages, returns = compute_rl4vlm_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            loss_mask=loss_mask,
            reward_mask=reward_mask,
            gamma=gamma,
            lam=lam,
        )
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    if adv_estimator == AdvantageEstimator.RL4VLM:
        think_mask = data.batch.get("think_mask", None)
        if think_mask is not None:
            data.batch["think_mask"] = think_mask[:, -response_length:]
        data.batch["reward_mask"] = reward_mask
        batch_size = advantages.shape[0]
        data.batch["thought_prob_coef"] = torch.full(
            (batch_size, 1),
            float(thought_prob_coef),
            dtype=torch.float32,
            device=advantages.device,
        )

        # RL4VLM learns one state value per turn, at the first generated token.
        first_response_mask = torch.zeros_like(loss_mask)
        for batch_idx in range(batch_size):
            end_positions = reward_mask[batch_idx].nonzero(as_tuple=True)[0]
            for turn_idx, end_position in enumerate(end_positions):
                start = 0 if turn_idx == 0 else end_positions[turn_idx - 1].item() + 1
                generated = loss_mask[batch_idx, start : end_position.item() + 1]
                local_positions = generated.nonzero(as_tuple=True)[0]
                position = (
                    start + local_positions[0].item()
                    if local_positions.numel() > 0
                    else end_position.item()
                )
                first_response_mask[batch_idx, position] = 1.0

        full_length = data.batch["attention_mask"].shape[1]
        full_first_response_mask = torch.zeros(
            batch_size,
            full_length,
            dtype=first_response_mask.dtype,
            device=first_response_mask.device,
        )
        full_first_response_mask[:, -response_length:] = first_response_mask
        data.batch["end_of_response_position_mask"] = full_first_response_mask
    return data


def get_current_turn_length(
    loss_mask: torch.Tensor,
    cur_position: int,
    batch_idx: int = 0,
) -> int:
    """
    Compute the length of the current turn based on loss_mask and reward_mask.

    Args:
        loss_mask (torch.Tensor): (bs, seq_len). 1 indicates token belongs to a valid turn.
        reward_mask (torch.Tensor): (bs, seq_len). 1 indicates EOS token (end of a turn).
        cur_position (int): Current token position (index).
        batch_idx (int): Which sequence in the batch to use. Default is 0.

    Returns:
        int: The length of the current turn.
    """
    # Check if current position is inside a valid turn
    if loss_mask[batch_idx, cur_position] == 0:
        raise ValueError(f"cur_position {cur_position} is not inside any turn.")

    # Step 1: Find the start position of the current turn
    # Move backward until we hit the first token of the turn
    start_pos = cur_position.clone().detach()

    while start_pos >= 0 and loss_mask[batch_idx, start_pos - 1] == 1:
        start_pos -= 1

    # Step 2: Find the EOS position of the current turn
    # Move forward until we find reward_mask == 1 (end of this turn)
    eos_pos = cur_position.clone().detach()
    seq_len = loss_mask.size(1)
    while eos_pos < seq_len and loss_mask[batch_idx, eos_pos] == 1:
        eos_pos += 1

 
        
    # Step 3: Compute the turn length (inclusive of both start and EOS)
    turn_length = eos_pos - start_pos

    return turn_length






def compute_rl4vlm_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    reward_mask: torch.Tensor,
    gamma: float,
    lam: float,
):
    """Compute RL4VLM turn-level GAE and broadcast it to generated tokens.

    Each interaction turn is one MDP step. V(s_t) is read at the first
    generated token of turn t, while r_t is read at that turn's boundary.
    """
    with torch.no_grad():
        batch_size, _ = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        for batch_idx in range(batch_size):
            end_positions = reward_mask[batch_idx].nonzero(as_tuple=True)[0]
            if end_positions.numel() == 0:
                continue

            first_response_positions = []
            for turn_idx, end_position in enumerate(end_positions):
                start = 0 if turn_idx == 0 else end_positions[turn_idx - 1].item() + 1
                generated = loss_mask[batch_idx, start : end_position.item() + 1]
                local_positions = generated.nonzero(as_tuple=True)[0]
                first_response_positions.append(
                    start + local_positions[0].item()
                    if local_positions.numel() > 0
                    else end_position.item()
                )

            turn_count = len(first_response_positions)
            turn_advantages = torch.zeros(
                turn_count,
                dtype=token_level_rewards.dtype,
                device=token_level_rewards.device,
            )
            last_gae = torch.zeros((), device=token_level_rewards.device)
            for turn_idx in range(turn_count - 1, -1, -1):
                current_value = values[batch_idx, first_response_positions[turn_idx]]
                next_value = (
                    values[batch_idx, first_response_positions[turn_idx + 1]]
                    if turn_idx < turn_count - 1
                    else torch.zeros_like(current_value)
                )
                reward = token_level_rewards[batch_idx, end_positions[turn_idx]]
                delta = reward + gamma * next_value - current_value
                last_gae = delta + gamma * lam * last_gae
                turn_advantages[turn_idx] = last_gae

            for turn_idx in range(turn_count):
                first_response = first_response_positions[turn_idx]
                returns[batch_idx, first_response] = (
                    turn_advantages[turn_idx] + values[batch_idx, first_response]
                )
                start = 0 if turn_idx == 0 else end_positions[turn_idx - 1].item() + 1
                end = end_positions[turn_idx].item() + 1
                generated = loss_mask[batch_idx, start:end].bool()
                advantages[batch_idx, start:end][generated] = turn_advantages[turn_idx]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    return advantages, returns


def compute_gae_advantage_return_with_loss_mask(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
):
    """Modified GAE calculation that handle multi-turn with loss mask
    Here we should also ensure that the trajectory score is given at the last valid token instead of last token
    Seems it's true in reward manager
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        for b in range(batch_size):
            lastgaelam = 0.0

            # Find the valid token positions (where loss_mask is 1)
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]

            if len(valid_positions) == 0:
                continue

            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]

                # Get the next value
                if i < len(valid_positions) - 1:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]

                else:
                    # Last valid position
                    nextvalue = 0.0

                # Calculate delta using the next valid token
                delta = (
                    token_level_rewards[b, curr_pos]
                    + gamma * nextvalue
                    - values[b, curr_pos]
                )

                # Update advantage estimate
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

            # Calculate returns for valid positions
            for i, pos in enumerate(valid_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)

    return advantages, returns


def compute_bi_level_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    reward_mask: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
    high_level_gamma: float,
):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        reward_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for reward position (end of each llm response)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()

        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]

                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]

                else:
                    # Last valid position
                    nextvalue = 0.0

                # Calculate delta using the next valid token
                delta = (
                    updated_reward[b, curr_pos]
                    + high_level_gamma * nextvalue
                    - values[b, curr_pos]
                )

                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if i < len(valid_positions) - 1:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = (
                    updated_reward[b, curr_pos]
                    + gamma * nextvalue
                    - values[b, curr_pos]
                )
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)

    return advantages, returns

def compute_hygae_return(
    token_level_rewards: torch.Tensor,
    reward_mask: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
    alpha: float = 0.5,
    low_level_lam: float = 1.0,
    aggregate_reward_in_turn: bool = False,
    whiten_advantage: bool = True,
):
    """Compute bi-level GAE with separate whitening, then mix.
    1. turn eos traj: use gae to compute adv and return, reward is the reward of the turn and     
   turn end position, accum_gamma = gamma^turn_len                                                                
    2. within turn: all tokens same adv                                               
    3. within turn: return[t] = reward[t] + gamma * return[t+1]

    Returns:
        advantages: (B, T) combined advantage after separate whitening.
        returns:    (B, T) combined return after separate whitening.
    """
    with torch.no_grad():
        B, T = token_level_rewards.shape

        adv_high = torch.zeros_like(token_level_rewards)  # will be broadcast to turn tokens
        ret_high = torch.zeros_like(token_level_rewards)  # will be broadcast to turn tokens
        adv_tok  = torch.zeros_like(token_level_rewards)
        ret_tok  = torch.zeros_like(token_level_rewards)

        updated_reward = token_level_rewards.clone()

        # ===== High-level GAE (computed at EOS, then broadcast to the entire turn) =====
        for b in range(B):
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            if eos_positions.numel() == 0:
                continue

            lastgaelam = 0.0
            # First, compute GAE only at EOS positions (your original design preserved)
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]

                # Value at next EOS (or 0.0 if this is the last turn)
                if i < len(eos_positions) - 1:
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    # Accumulate token-scale discount across the current turn (kept as in your design)
                    turn_len = get_current_turn_length(
                        loss_mask=loss_mask, 
                        cur_position=next_pos, batch_idx=b,
                    )
                else:
                    next_pos = curr_pos
                    nextvalue = 0.0
                    turn_len = 1
                
                accum_gamma = gamma ** turn_len

                # GAE delta at EOS (turn-level)
                if aggregate_reward_in_turn and next_pos != curr_pos:
                    # aggregate reward in from the current position to the end of the turn
                    valid_positions_for_aggregate = loss_mask[b, curr_pos+1:next_pos].nonzero(as_tuple=True)[0] + curr_pos + 1
                
                    reward = updated_reward[b, curr_pos].clone()
                    factor = 1
                    for pp in valid_positions_for_aggregate:
                        reward += gamma ** factor * updated_reward[b, pp]
                        factor += 1
                else:
                    reward = updated_reward[b, curr_pos]

                # GAE delta at EOS (turn-level)
                delta = (
                    reward
                    + accum_gamma * nextvalue
                    - values[b, curr_pos]
                )
                lastgaelam = delta + accum_gamma * lam * lastgaelam
                

                # Temporarily store only at EOS; we'll broadcast below
                adv_high[b, curr_pos] = lastgaelam
                ret_high[b, curr_pos] = lastgaelam + values[b, curr_pos]

            # Now BROADCAST each EOS's turn-level value to all tokens in that turn.
            # A "turn" is defined as tokens (loss_mask==1) between the previous EOS (exclusive)
            # and the current EOS (inclusive).
            for i, curr_pos in enumerate(eos_positions):
                prev_eos = eos_positions[i - 1].item() if i > 0 else -1
                # Valid token range for this turn is (prev_eos, curr_pos], but restricted by loss_mask
                turn_slice = slice(prev_eos + 1, curr_pos + 1)
                # Create a boolean mask for the tokens belonging to this turn
                turn_token_mask = torch.zeros(T, dtype=torch.bool, device=loss_mask.device)
                # Only broadcast to positions that are valid LLM tokens
                turn_token_mask[turn_slice] = loss_mask[b, turn_slice].bool()

                # Broadcast the EOS-computed advantage/return to all tokens of the turn
                adv_val = adv_high[b, curr_pos].clone()
                adv_high[b, turn_token_mask] = adv_val
            for i, curr_pos in enumerate(eos_positions):
                prev_eos = eos_positions[i - 1].item() if i > 0 else -1
                # (prev_eos, curr_pos] 是这个 turn 的 token 区间
                base_indices = torch.arange(
                    prev_eos + 1,
                    curr_pos + 1,
                    device=loss_mask.device,
                )

                turn_token_indices = base_indices[loss_mask[b, base_indices].bool()]
  
                for k in range(turn_token_indices.numel() - 2, -1, -1):
                    t = turn_token_indices[k]
                    t_next = turn_token_indices[k + 1]
                    ret_high[b, t] = (
                        updated_reward[b, t]
                        + gamma * ret_high[b, t_next]
                    )


        # ===== Token-level GAE (standard over valid positions) =====
        for b in range(B):
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            if valid_positions.numel() == 0:
                continue
            lastgaelam = 0.0
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if i != len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    nextvalue = 0.0
                    lastgaelam = 0.0

                delta = (
                    updated_reward[b, curr_pos]
                    + gamma * nextvalue
                    - values[b, curr_pos]
                )
                lastgaelam = delta + gamma * low_level_lam * lastgaelam
                
                adv_tok[b, curr_pos] = lastgaelam
                ret_tok[b, curr_pos] = lastgaelam + values[b, curr_pos]

        # ===== Mix the two signals =====
        advantages = alpha * adv_high + (1.0 - alpha) * adv_tok
        returns    = alpha * ret_high + (1.0 - alpha) * ret_tok
        if whiten_advantage:
            advantages = verl_F.masked_whiten(advantages, loss_mask)

    return advantages, returns


def compute_turn_wise_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    reward_mask: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    lam: float,
    high_level_gamma: float,
):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        reward_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for reward position (end of each llm response)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]

                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]

                else:
                    # Last valid position
                    nextvalue = 0.0

                # Calculate delta using the next valid token
                delta = (
                    token_level_rewards[b, curr_pos]
                    + high_level_gamma * nextvalue
                    - values[b, curr_pos]
                )

                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]

            # each token in the sequence has the same advantage
            cur_adv = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    advantages[b, curr_pos] = cur_adv
                else:
                    # Last valid position
                    cur_adv = advantages[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, reward_mask)

    return advantages, returns


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = (
                token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            )  # TD error
            lastgaelam = delta + gamma * lam * lastgaelam  # gae
            advantages_reversed.append(lastgaelam)  # store the gae
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_masked_reinforce_advantage(
    token_level_rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    Compute advantage for masked REINFORCE.
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
        gamma: `(torch.Tensor)`
            shape: (bs, response_length)
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        for b in range(batch_size):
            running_return = 0
            # Find the valid token positions (where loss_mask is 1)
            valid_positions = (loss_mask[b] == 1).nonzero(as_tuple=True)[0]

            if len(valid_positions) == 0:
                continue

            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                running_return = token_level_rewards[b, curr_pos] + gamma * running_return
                advantages[b, curr_pos] = running_return
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    return advantages, torch.zeros_like(token_level_rewards)


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    eos_mask: torch.Tensor,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (
            (token_level_rewards * eos_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = (
            returns
            - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask
        )

    return advantages, returns

