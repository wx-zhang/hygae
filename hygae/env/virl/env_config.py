from hygae.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from typing import Optional

@dataclass
class VIRLEnvConfig(BaseEnvConfig):
    """Configuration for the self-contained, offline VIRL environment."""
    env_name: str = "virl"
    route_info_path: str = ""
    relocation: bool = False
    drop_rate: float = 0.0
    straight_line_length: int = 2
    resolution: int = 1200
    verify_iter: int = 0
    language_only: bool = False
    absolute_action: bool = True
    max_actions_per_step: int = 3
    format_reward: float = 0.3
    success_reward: float = 10.0
    keypoint_reward: float = 3.0
    action_correct_reward: float = 0.8
    prompt_format: str = "free_think"
    panorama_dir: Optional[str] = None
    gps_to_pano_path: Optional[str] = None
    mapping_radius: float = 50.0
    intersection_valid_radius: float = 10.0
    oracle_radius: float = 20.0
    # Legacy nested fields are accepted so experiment YAMLs remain reusable.
    platform_cfg: Optional[dict] = None
    navigator_cfg: Optional[dict] = None

    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["route_info_path", "absolute_action", "language_only", "max_actions_per_step"]
        id_str = ",".join([f"{f.name}={getattr(self, f.name)}" for f in fields(self) if f.name in id_fields])
        return f"VIRLEnvConfig({id_str})"
