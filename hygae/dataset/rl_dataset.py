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
RL dataset for multi-turn agent training.

This dataset provides environment configurations that are passed to the agent loop
via `extra_info` in `non_tensor_batch`. The agent loop uses these configs to
initialize and interact with environments.

Usage:
    dataset = RLDataset(
        env_configs=[
            {"env_name": "my_env", "env_config": {...}, "seed": 42},
            ...
        ],
        tokenizer=tokenizer,
    )
"""

import copy
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class RLDataset(Dataset):
    """
    Dataset for RL training with environment-based agent loops.
    
    This dataset provides environment configurations that are passed to the
    training agent loop. Each sample contains:
    - `extra_info`: Environment configuration dict with `env_name`, `env_config`, `seed`
    - `raw_prompt`: Initial prompt messages (can be empty for agent loop)
    - Other fields required by verl framework
    
    The configured agent loop receives these via kwargs and uses them to
    create and interact with environments.
    
    Args:
        env_configs: List of environment configuration dictionaries. Each should contain:
            - env_name (str): Name of the registered environment
            - env_config (dict): Environment-specific configuration
            - seed (int, optional): Random seed for environment
        tokenizer: Tokenizer for encoding prompts
        processor: Optional processor for multi-modal data
        repeat_times: Number of times to repeat each config (for multiple rollouts)
        shuffle_seed: Random seed for shuffling (None for no shuffle)
        default_seed_range: Range for generating random seeds if not provided
    """
    
    def __init__(
        self,
        env_configs: List[Dict[str, Any]],
        tokenizer,
        processor=None,
        repeat_times: int = 1,
        shuffle_seed: Optional[int] = None,
        default_seed_range: tuple = (0, 100000),
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.repeat_times = repeat_times
        self.default_seed_range = default_seed_range
        
        # Process and validate env_configs
        self.env_configs = self._process_configs(env_configs)
        
        # Optionally shuffle
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            rng.shuffle(self.env_configs)
    
    def _process_configs(self, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and validate environment configurations.
        
        Args:
            env_configs: Raw environment configurations
            
        Returns:
            Processed list of configurations with required fields
        """
        processed = []
        
        for cfg in env_configs:


            # Create processed config
            processed_cfg = cfg["extra_info"]
            

            # Repeat if needed
            for i in range(self.repeat_times):
                cfg_copy = copy.deepcopy(processed_cfg)
                # Vary seed for each repeat
                if self.repeat_times > 1:
                    cfg_copy["seed"] = processed_cfg["seed"] + i
                processed.append(cfg_copy)
        
        return processed
    
    def __len__(self) -> int:
        return len(self.env_configs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample for the agent loop.
        
        Returns a dictionary containing:
        - extra_info: Environment configuration for agent loop
        - raw_prompt: Initial prompt messages (empty list for agent loop to fill)
        - input_ids: Placeholder tensor
        - attention_mask: Placeholder tensor
        - Other fields as needed
        """
        env_config = self.env_configs[idx]
        
        # Create the sample dictionary
        # The agent loop will use extra_info to create the environment
        sample = {
            # Environment configuration passed to agent loop
            "extra_info": env_config,
            
            # Empty raw_prompt - agent loop will build conversation from environment
            "raw_prompt": [],
            
            # Placeholder tensors (agent loop handles actual tokenization)
            "input_ids": torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
            "attention_mask": torch.tensor([0], dtype=torch.long),
            "position_ids": torch.tensor([0], dtype=torch.long),
            
            # Data source identifier
            "data_source": env_config.get("env_name", "hygae"),
            

        }
        
        return sample


class RLDatasetFromFile(RLDataset):
    """
    Load environment configurations from JSON/JSONL/Parquet files.

    File format (JSON):
    [
        {"env_name": "my_env", "env_config": {...}, "seed": 42},
        ...
    ]

    File format (JSONL):
    {"env_name": "my_env", "env_config": {...}, "seed": 42}
    {"env_name": "my_env", "env_config": {...}, "seed": 43}
    ...

    File format (Parquet):
    Parquet file with columns: env_name, env_config, seed (optional), etc.

    Args:
        data_files: Path or list of paths to data files
        tokenizer: Tokenizer for encoding
        processor: Optional processor for multi-modal data
        repeat_times: Number of times to repeat each config
        shuffle_seed: Random seed for shuffling
        max_samples: Maximum number of samples to load (-1 for all)
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer,
        processor=None,
        repeat_times: int = 1,
        shuffle_seed: Optional[int] = None,
        max_samples: int = -1,
    ):
        import json
        from pathlib import Path

        if isinstance(data_files, str):
            data_files = [data_files]

        env_configs = []

        for file_path in data_files:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            if path.suffix == ".parquet":
                # Parquet format: use HuggingFace datasets library
                import datasets
                dataframe = datasets.load_dataset("parquet", data_files=str(path))["train"]
                # Convert dataset to list of dicts
                for i in range(len(dataframe)):
                    env_configs.append(dict(dataframe[i]))
            elif path.suffix == ".jsonl":
                # JSONL format: one JSON object per line
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            env_configs.append(json.loads(line))
            elif path.suffix == ".json":
                # JSON format: list of objects
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        env_configs.extend(data)
                    else:
                        env_configs.append(data)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats: .parquet, .json, .jsonl")

        # Apply max_samples limit
        if max_samples > 0 and len(env_configs) > max_samples:
            env_configs = env_configs[:max_samples]

        super().__init__(
            env_configs=env_configs,
            tokenizer=tokenizer,
            processor=processor,
            repeat_times=repeat_times,
            shuffle_seed=shuffle_seed,
        )


def collate_fn(batch: List[Dict[str, Any]]) -> dict:
    """
    Collate function for RLDataset.

    Batches samples by stacking tensors and collecting non-tensors into numpy arrays.
    Returns a plain dictionary that the trainer will convert to DataProto.

    Args:
        batch: List of sample dictionaries

    Returns:
        Dict where tensor entries are stacked into torch.Tensor of shape (batch_size, *dims)
        and non-tensor entries are converted to np.ndarray of dtype object with shape (batch_size,)
    """
    if not batch:
        return {}

    # Separate tensor and non-tensor keys
    tensor_dict = {}
    non_tensor_dict = {}

    first_sample = batch[0]
    for key, value in first_sample.items():
        if isinstance(value, torch.Tensor):
            # Stack tensors with padding
            tensors = [sample[key] for sample in batch]

            # Find max length
            max_len = max(t.shape[0] for t in tensors)

            # Pad and stack
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    padding = torch.full(
                        (max_len - t.shape[0],) + t.shape[1:],
                        0,  # pad with zeros
                        dtype=t.dtype,
                    )
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)

            tensor_dict[key] = torch.stack(padded, dim=0)
        else:
            # Collect non-tensor data into numpy arrays
            values = [sample[key] for sample in batch]
            non_tensor_dict[key] = np.array(values, dtype=object)

    # Return plain dict - trainer will convert to DataProto
    return {**tensor_dict, **non_tensor_dict}


# Convenience function to create dataset from config
def create_rl_dataset(
    config,
    tokenizer,
    processor=None,
    split: str = "train",
) -> RLDataset:
    """
    Create an RLDataset from configuration.
    
    Args:
        config: Configuration object with data settings
        tokenizer: Tokenizer for encoding
        processor: Optional processor for multi-modal
        split: "train" or "val"
        
    Returns:
        RLDataset instance
    """
    if split == "train":
        data_files = config.data.train_files
        max_samples = config.data.get("train_max_samples", -1)
        shuffle_seed = config.data.get("seed", 42)
    else:
        data_files = config.data.val_files
        max_samples = config.data.get("val_max_samples", -1)
        shuffle_seed = None  # Don't shuffle validation
    
    repeat_times = config.data.get("repeat_times", 1)
    
    return RLDatasetFromFile(
        data_files=data_files,
        tokenizer=tokenizer,
        processor=processor,
        repeat_times=repeat_times,
        shuffle_seed=shuffle_seed,
        max_samples=max_samples,
    )
