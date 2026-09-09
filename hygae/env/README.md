# HYGAE environments

HYGAE exposes five environments through a lazy registry. Installing or importing one
environment does not import the optional dependencies of another.

| Name | Extra setup | Dataset config |
| --- | --- | --- |
| `sokoban` | Included in the base installation | `configs/sokoban_env_config.yaml` |
| `frozenlake` | Included in the base installation | `configs/frozenlake_env_config.yaml` |
| `virl` | Set `VIRL_DATA_ROOT` to the offline panorama dataset | `configs/virl_env_config.yaml` |
| `navigation` | Install with `--with-navigation`; uses AI2-THOR cloud rendering | `configs/navigation_env_config.yaml` |
| `primitive_skill` | Install with `--with-primitive-skill`; requires ManiSkill assets | `configs/primitive_skill_env_config.yaml` |

Primitive Skill uses the GPU renderer by default. Set
`HYGAE_PRIMITIVE_RENDER_BACKEND=cpu` to use its CPU wrapper. On headless machines,
configure a working Vulkan ICD before starting the environment.

See the repository-level `README.md` for installation, dataset generation, and training
commands.
