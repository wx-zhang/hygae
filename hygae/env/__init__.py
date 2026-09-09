"""Lazy registry for the environments shipped in the public HYGAE release."""

from __future__ import annotations

import importlib


REGISTERED_ENV: dict[str, dict[str, type]] = {}

_ENV_MODULES = {
    "sokoban": ("hygae.env.sokoban", "SokobanEnv", "SokobanEnvConfig"),
    "frozenlake": ("hygae.env.frozenlake", "FrozenLakeEnv", "FrozenLakeEnvConfig"),
    "navigation": ("hygae.env.navigation", "NavigationEnv", "NavigationEnvConfig"),
    "primitive_skill": (
        "hygae.env.primitive_skill",
        "PrimitiveSkillEnv",
        "PrimitiveSkillEnvConfig",
    ),
    "virl": ("hygae.env.virl", "VIRLEnv", "VIRLEnvConfig"),
}


def register_env(env_name: str) -> None:
    """Import and register one of the five paper environments on demand."""

    if env_name in REGISTERED_ENV:
        return
    try:
        module_name, env_class_name, config_class_name = _ENV_MODULES[env_name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENV_MODULES))
        raise ValueError(
            f"Unknown environment {env_name!r}. Available environments: {available}"
        ) from exc

    module = importlib.import_module(module_name)
    REGISTERED_ENV[env_name] = {
        "env_cls": getattr(module, env_class_name),
        "config_cls": getattr(module, config_class_name),
    }


def available_envs() -> tuple[str, ...]:
    """Return the stable public environment identifiers."""

    return tuple(sorted(_ENV_MODULES))
