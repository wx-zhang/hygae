"""Helpers for attaching environment-state metadata to local rollouts.

The release trainer does not call an LLM judge.  Environments may still expose
pre/post state strings so a caller can compute deterministic process metrics.
"""

from functools import wraps


def env_state_reward_wrapper(step_func):
    """Capture state metadata around a step when ``use_state_reward`` is set."""

    @wraps(step_func)
    def wrapped_step(self, action_str):
        if not (
            hasattr(self, "config")
            and self.config.get("use_state_reward", False)
        ):
            return step_func(self, action_str)

        prompt_format = self.config.get("prompt_format")
        if prompt_format is None:
            raise ValueError("prompt_format is required when use_state_reward=True")
        if "grounding" not in prompt_format and "worldmodeling" not in prompt_format:
            raise ValueError(
                "use_state_reward requires a grounding or worldmodeling prompt format"
            )

        pre_state = self.get_env_state()
        observation, reward, done, info = step_func(self, action_str)
        post_state = self.get_env_state()

        metrics = info.setdefault("metrics", {})
        turn_metrics = metrics.setdefault("turn_metrics", {})
        metrics.setdefault("traj_metrics", {})
        if info.get("is_format_rewarded", False):
            info["use_state_reward"] = True
            if info.get("observation_content"):
                info["observation_state"] = pre_state
            if info.get("prediction_content"):
                info["prediction_state"] = post_state
        else:
            info["use_state_reward"] = False
            if info.get("observation_content"):
                turn_metrics["grounding_reward"] = 0.0
            if info.get("prediction_content"):
                turn_metrics["worldmodeling_reward"] = 0.0
        return observation, reward, done, info

    return wrapped_step
