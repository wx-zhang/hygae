# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with trainer as trainer is used by other main.
"""
import hydra

import ray
from hygae.trainer.ppo.trainer import Trainer
from hygae.utils.compute_score import compute_score


@hydra.main(config_path="config", config_name="trainer", version_base=None)
def main(config):
    run_ppo(config, compute_score=compute_score)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            }
        )

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf
    from verl.utils.fs import copy_to_local

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(
        local_path, use_fast=True
    )  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        try:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
        except ImportError:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker as AsyncActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        try:
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker
        except ImportError:
            from verl.workers.megatron_workers import ActorRolloutRefWorker as AsyncActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from hygae.trainer.ppo.trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    use_ref = config.actor_rollout_ref.ref.get("use_ref", True)
    if use_ref:
        role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncActorRolloutRefWorker)
    else:
        config.actor_rollout_ref.actor.use_kl_loss = False
        print("[WARNING] Ref policy is disabled, use_kl_loss is set to False")

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if use_ref:
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import (
                GenerativeRewardWorker,
                RewardModelWorker,
            )
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = (
            ray.remote(RewardModelWorker)
            if config.reward_model.model_type == "sequence_classification"
            else ray.remote(GenerativeRewardWorker)
        )
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError
    reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=0, compute_score=compute_score
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=1, compute_score=compute_score
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    trainer = Trainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
