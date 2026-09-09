#!/usr/bin/env python3
"""Fail-fast verification for the supported HYGAE Conda installation."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERL_BASE = "329dcfe1dd60f2d736ee55914e2a49e1887718eb"
EXPECTED_VERSIONS = {
    "accelerate": "1.12.0",
    "codetiming": "1.4.0",
    "datasets": "4.6.0",
    "dill": "0.4.0",
    "fsspec": "2026.2.0",
    "gym": "0.26.2",
    "gym-sokoban": "0.0.6",
    "gymnasium": "0.29.1",
    "hydra-core": "1.3.2",
    "math-verify": "0.9.0",
    "matplotlib": "3.10.8",
    "ninja": "1.13.0",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.11.0.86",
    "openai": "2.24.0",
    "pandas": "2.3.3",
    "packaging": "25.0",
    "peft": "0.18.1",
    "pillow": "12.1.1",
    "pyarrow": "23.0.1",
    "pybind11": "3.0.2",
    "pygame": "2.6.1",
    "pylatexenc": "2.10",
    "pyyaml": "6.0.3",
    "qwen-vl-utils": "0.0.14",
    "ray": "2.54.0",
    "tensordict": "0.6.2",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "torchdata": "0.11.0",
    "torchvision": "0.21.0",
    "tqdm": "4.67.3",
    "transformers": "4.49.0",
    "tokenizers": "0.21.4",
    "triton": "3.2.0",
    "vllm": "0.8.2",
    "wandb": "0.25.0",
    "xformers": "0.0.29.post2",
    "xgrammar": "0.1.16",
    "flash-attn": "2.7.4.post1",
}
OPTIONAL_EXPECTED_VERSIONS = {
    "ai2thor": "5.0.0",
    "mani-skill": "3.0.0b20",
    "numba": "0.60.0",
    "sapien": "3.0.0b1",
    "transforms3d": "0.4.2",
}

FLASH_ATTN_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)
FLASH_ATTN_SHA256 = "ffe17686fa1a0f288de9eae7c32af209d32a27b037ef28614f042b377af5b15a"


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"HYGAE requires Python 3.10, found {sys.version}")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise SystemExit("no active Conda environment detected")
    if Path(sys.prefix).resolve() != Path(conda_prefix).resolve():
        raise SystemExit(
            f"Python prefix {sys.prefix} does not match CONDA_PREFIX {conda_prefix}"
        )

    head = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/verl"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != VERL_BASE:
        raise SystemExit(f"wrong VERL base: {head}")

    versions = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise SystemExit(
            "runtime version mismatch:\n"
            f"actual={versions}\n"
            f"expected={EXPECTED_VERSIONS}"
        )

    for name, expected in OPTIONAL_EXPECTED_VERSIONS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        if actual != expected:
            raise SystemExit(
                f"optional dependency version mismatch for {name}: "
                f"actual={actual}; expected={expected}"
            )

    direct_url_text = metadata.distribution("flash-attn").read_text("direct_url.json")
    if not direct_url_text:
        raise SystemExit("flash-attn is not installed from the verified wheel")
    direct_url = json.loads(direct_url_text)
    actual_url = direct_url.get("url")
    if actual_url != FLASH_ATTN_URL:
        raise SystemExit(
            f"wrong flash-attn wheel URL: {actual_url}; expected {FLASH_ATTN_URL}"
        )
    actual_hash = direct_url.get("archive_info", {}).get("hash", "").removeprefix(
        "sha256="
    )
    if actual_hash != FLASH_ATTN_SHA256:
        raise SystemExit(
            f"wrong flash-attn wheel hash: {actual_hash}; expected {FLASH_ATTN_SHA256}"
        )

    import torch
    import hygae
    import verl
    from hygae.env import available_envs
    from verl.utils.value_head_init import DEFAULT_VALUE_HEAD_INIT_STD, VALUE_HEAD_INIT_SEED

    if ROOT not in Path(hygae.__file__).resolve().parents:
        raise SystemExit(f"hygae imported from wrong checkout: {hygae.__file__}")
    if ROOT / "third_party/verl" not in Path(verl.__file__).resolve().parents:
        raise SystemExit(f"verl imported from wrong checkout: {verl.__file__}")
    if available_envs() != (
        "frozenlake",
        "navigation",
        "primitive_skill",
        "sokoban",
        "virl",
    ):
        raise SystemExit(f"unexpected environment registry: {available_envs()}")
    if VALUE_HEAD_INIT_SEED != 42 or DEFAULT_VALUE_HEAD_INIT_STD != 0.002:
        raise SystemExit("unexpected critic value-head initialization defaults")

    print("HYGAE preflight: OK")
    print(f"Conda prefix: {conda_prefix}")
    print(f"VERL base: {VERL_BASE} + installation/verl.patch")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}; GPUs: {torch.cuda.device_count()}")


if __name__ == "__main__":
    main()
