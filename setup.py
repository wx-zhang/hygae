from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="hygae",
    version="0.2.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "hygae": [
            "trainer/config/*.yaml",
            "env/navigation/datasets/*.json",
        ]
    },
    python_requires=">=3.10,<3.11",
    description="Hybrid Advantage Estimation for VLM agentic reinforcement learning",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=[
        "accelerate==1.12.0",
        "codetiming==1.4.0",
        "datasets==4.6.0",
        "dill==0.4.0",
        "fsspec==2026.2.0",
        "gym==0.26.2",
        "gym-sokoban==0.0.6",
        "gymnasium==0.29.1",
        "hydra-core==1.3.2",
        "math-verify==0.9.0",
        "matplotlib==3.10.8",
        "ninja==1.13.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
        "openai==2.24.0",
        "pandas==2.3.3",
        "packaging==25.0",
        "peft==0.18.1",
        "pillow==12.1.1",
        "pyarrow==23.0.1",
        "pybind11==3.0.2",
        "pygame==2.6.1",
        "pylatexenc==2.10",
        "pyyaml==6.0.3",
        "qwen-vl-utils==0.0.14",
        "ray[default]==2.54.0",
        "tensordict==0.6.2",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchdata==0.11.0",
        "torchvision==0.21.0",
        "tqdm==4.67.3",
        "tokenizers==0.21.4",
        "transformers==4.49.0",
        "triton==3.2.0",
        "vllm==0.8.2",
        "wandb==0.25.0",
        "xformers==0.0.29.post2",
        "xgrammar==0.1.16",
    ],
    extras_require={
        "navigation": ["ai2thor==5.0.0"],
        "primitive-skill": [
            "mani-skill==3.0.0b20",
            "numba==0.60.0",
            "sapien==3.0.0b1",
            "transforms3d==0.4.2",
        ],
    },
)
