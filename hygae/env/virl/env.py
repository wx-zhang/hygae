import json
import random
from PIL import Image
from typing import Tuple, Dict, Optional

from hygae.env.base.base_env import BaseEnv
from hygae.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import VIRLEnvConfig
from .prompt import system_prompt as virl_system_prompt
from .prompt import init_observation_template, action_template, format_prompt
from .geospatial import distance_meters
from .panorama import OfflinePanoramaStore
from .parsing import parse_navigation_string


class VIRLEnv(BaseEnv):
    """VIRL navigation environment adapted from the SFTvsRL route benchmark.

    It uses only the published offline panorama/mapping files.  No Google API,
    browser driver, SFTvsRL checkout, or network fallback is involved.
    """

    VALID_ACTIONS = ["forward()", "stop()"]
    ORIENTATION_SET = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    ORIENTATION_HEADING = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, config: VIRLEnvConfig):
        super().__init__()
        self.config = config

        # Load route data
        with open(config.route_info_path) as f:
            route_data = json.load(f)
        self.route_list = route_data[0]
        self.route_num = route_data[1]

        self.panorama_store = OfflinePanoramaStore.from_env_config(config)
        landmark_cfg = (config.navigator_cfg or {}).get("LANDMARK_DETECT", {})
        self.intersection_valid_radius = float(
            landmark_cfg.get("INTERSECTION_VALID_RADIUS", config.intersection_valid_radius)
        )
        self.oracle_radius = float(landmark_cfg.get("ORACLE_RADIUS", config.oracle_radius))
        self.resolution = config.resolution
        self.relocation = config.relocation
        self.drop_rate = config.drop_rate
        self.straight_line_length = config.straight_line_length
        self.language_only = config.language_only
        self.absolute_action = config.absolute_action

        # Setup parse function and format prompt
        self.parse_func = PARSE_FUNC_MAP[config.prompt_format]
        self.format_prompt_func = format_prompt[config.prompt_format]

        # Episode state
        self.gt_rail_info = []
        self.step_cnt = 0
        self.current_geocode = None
        self.current_heading = 0
        self.target_heading = 0
        self.action_list = []
        self.observation_list = []
        self.traj_list = []
        self.str_instruction = ""
        self.list_instruction = []
        self.instruction_idx = 0
        self.total_reward = 0
        self.reward = 0
        self.valid_actions = []
        self.info = {}

        # Rail-based navigation state
        self.reached_instructions = set()
        self.instruction_keypoints = []
        self.destination_geocode = None
        self.rail_idx = 0

    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        if seed is not None:
            route_idx = seed % self.route_num
            self.route_info = self.route_list[route_idx]
        else:
            self.route_info = random.choice(self.route_list)

        start_place = self.route_info['start_place']
        start_position = start_place['relocated_geocode']

        self.current_geocode = (
            self.panorama_store.relocate(tuple(start_position))[0]
            if self.relocation
            else tuple(start_position)
        )

        self.current_heading = self.route_info.get('init_heading', 0)
        if self.current_heading == 0:
            if seed is not None:
                rng = random.Random(seed)
                self.current_heading = rng.randint(0, 359)
            else:
                self.current_heading = random.randint(0, 359)
        self.target_heading = self.current_heading
        self.landmark_list = self.route_info['route_results']['landmark_list'] + [self.route_info['dest_place']]

        self.action_list = []
        self.observation_list = []
        self.traj_list = [self.current_geocode]
        self.total_reward = 0
        self.reward = 0
        self.valid_actions = []

        # Parse instructions and build rail (still used for reference/keypoints)
        self.str_instruction, self.list_instruction = self._parse_instruction_and_rail()
        self.instruction_idx = 0
        self.step_cnt = 0

        # Rail-based navigation state
        self.reached_instructions = set()
        self._build_instruction_keypoints()
        self.destination_geocode = self.gt_rail_info[-1]['geocode']
        self.rail_idx = 0  # Track position on ground truth rail
        # Get initial observation
        obs = self._render(init_obs=True)

        info = {
            'global_instruction': self.str_instruction,
            'current_instruction': self.list_instruction[0] if self.list_instruction else "",
            'instruction_idx': 0,
        }

        return obs, info

    def step(self, action_str: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute rail-based navigation with action verification.

        Only executes the action if it matches the ground truth action at the
        current rail position. Otherwise, the agent stays in place and receives
        feedback that the action is incorrect.
        """
        # Parse LLM response
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 1)
        )

        action_list = rst['actions']
        format_correct = rst.get('format_correct', False)

        self.reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(rst)

        n_actions = 0
        n_correct_actions = 0
        action_correct = True
        all_keypoint_feedback = []
        n_reached_before = len(self.reached_instructions)

        if action_list and format_correct:
            for action in action_list:
                action = action.strip()
                self.valid_actions.append(action)
                n_actions += 1

                # Verify action against ground truth rail
                if self.rail_idx >= len(self.gt_rail_info):
                    action_correct = False
                    break

                expected_action = self.gt_rail_info[self.rail_idx]['gt_action']
                if action != expected_action:
                    # Incorrect action: do not execute, do not move
                    action_correct = False
                    break

                # Correct action: execute on rail
                n_correct_actions += 1
                if "stop()" in action:
                    done = True
                    self.action_list.append(action)
                    # Check all keypoints at stop position
                    kp_feedback = self._check_keypoints()
                    if kp_feedback:
                        all_keypoint_feedback.append(kp_feedback)
                    break
                else:
                    # For turn actions, check keypoints BEFORE advancing position
                    # because the keypoint geocode is at the turn step itself,
                    # but after rail_idx advances, the position jumps to the next
                    # step which may be >10m away (interpolated forward step).
                    if action.startswith('turn_direction'):
                        direction = action.split('(')[1].split(')')[0]
                        if direction in self.ORIENTATION_SET:
                            idx = self.ORIENTATION_SET.index(direction)
                            self.current_heading = self.ORIENTATION_HEADING[idx]
                            self.target_heading = self.current_heading
                        kp_feedback = self._check_keypoints()
                        if kp_feedback:
                            all_keypoint_feedback.append(kp_feedback)

                    # Advance on rail
                    self.rail_idx += 1
                    if self.rail_idx < len(self.gt_rail_info):
                        self.current_geocode = self.gt_rail_info[self.rail_idx]['geocode']
                        self.current_heading = self.gt_rail_info[self.rail_idx]['heading']
                        self.target_heading = self.current_heading

                    self.traj_list.append(tuple(self.current_geocode))
                    self.action_list.append(action)

                    # Check keypoints after advancing (for forward/all actions)
                    kp_feedback = self._check_keypoints()
                    if kp_feedback:
                        all_keypoint_feedback.append(kp_feedback)

        self.step_cnt += 1
        keypoint_feedback = " ".join(all_keypoint_feedback)

        # Determine success: destination keypoint reached
        dest_keypoint_idx = None
        for i, kp in enumerate(self.instruction_keypoints):
            if kp['is_destination']:
                dest_keypoint_idx = i
                break
        success = dest_keypoint_idx is not None and dest_keypoint_idx in self.reached_instructions

        # Per-instruction completion metrics
        n_total = len(self.instruction_keypoints)
        n_reached = len(self.reached_instructions)
        completed_instr = sorted([
            self.instruction_keypoints[kp_idx]['instr_idx']
            for kp_idx in self.reached_instructions
        ])

        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0 and format_correct,
                "action_is_effective": n_actions > 0 and action_correct,
                "n_actions": n_actions,
            },
            "traj_metrics": {
                "success": success,
                "progress": n_reached / max(n_total, 1),
                "n_keypoints_reached": n_reached,
                "n_keypoints_total": n_total,
                "completed_instructions": completed_instr,
            }
        }

        # Format reward
        if format_correct:
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False

        # Action correct reward: +1 per correct action
        if n_correct_actions > 0:
            self.reward += n_correct_actions * self.config.action_correct_reward

        # Keypoint reward: +2 per newly reached keypoint
        n_new_keypoints = len(self.reached_instructions) - n_reached_before
        if n_new_keypoints > 0:
            self.reward += n_new_keypoints * self.config.keypoint_reward

        # Success reward: +10 if destination reached
        if success:
            self.reward += self.config.success_reward

        # Build env feedback
        if not metrics["turn_metrics"]["action_is_valid"]:
            env_feedback = "Invalid format or no action detected."
        elif not action_correct:
            env_feedback = "This is not correct action."
        elif done and success:
            env_feedback = "Navigation complete! You reached the destination."
        elif done and not success:
            env_feedback = "You stopped but have not reached the destination."
        elif keypoint_feedback:
            env_feedback = f"Executed {n_actions} action(s). {keypoint_feedback}"
        else:
            env_feedback = f"Executed {n_actions} action(s)."

        info["env_feedback"] = env_feedback
        info["metrics"] = metrics
        info['instruction'] = self.str_instruction
        info['is_success'] = success

        # Update total reward
        self.total_reward += self.reward
        self.info = info

        obs = self._render(init_obs=False)
        return obs, self.reward, done, info

    def _build_instruction_keypoints(self):
        """Build instruction keypoints from gt_rail_info.

        Each instruction maps to a (geocode, heading) pair on the 2D plane.
        The agent must reach the correct coordinate AND face the correct direction
        to complete an instruction.

        Each keypoint has an action_type ('turn' or 'forward'):
          - Turn keypoints use the turn's own geocode + target heading,
            checked only after turn actions (with tight KEYPOINT_RADIUS).
          - Forward keypoints use the forward's endpoint geocode + travel heading,
            checked only after forward actions (with INTERSECTION_VALID_RADIUS).
        This prevents a forward() that happens to go east from falsely
        completing a 'turn east' instruction.
        """
        self.instruction_keypoints = []
        if not self.gt_rail_info:
            return

        # Find last rail step per instruction_idx
        last_step_per_instr = {}
        for step_info in self.gt_rail_info:
            instr_idx = step_info['instruction_idx']
            last_step_per_instr[instr_idx] = step_info

        for instr_idx in sorted(last_step_per_instr.keys()):
            step_info = last_step_per_instr[instr_idx]
            is_destination = (step_info['gt_action'] == 'stop()')

            if step_info['gt_action'].startswith('turn_direction'):
                direction = step_info['gt_action'].split('(')[1].split(')')[0]
                heading = self.ORIENTATION_HEADING[self.ORIENTATION_SET.index(direction)]
                action_type = 'turn'
            else:
                heading = step_info['heading']
                action_type = 'forward'

            self.instruction_keypoints.append({
                'geocode': step_info['geocode'],
                'heading': heading,
                'instr_idx': instr_idx,
                'action_type': action_type,
                'is_destination': is_destination,
            })

    HEADING_TOLERANCE = 45  # degrees

    @staticmethod
    def _heading_diff(h1, h2):
        """Minimum angle between two headings (0-360)."""
        diff = abs(h1 - h2) % 360
        return min(diff, 360 - diff)

    def _heading_to_direction(self, heading):
        """Convert a heading (0-360) to the nearest compass direction name."""
        heading = heading % 360
        min_diff = 360
        best = self.ORIENTATION_SET[0]
        for name, h in zip(self.ORIENTATION_SET, self.ORIENTATION_HEADING):
            d = self._heading_diff(heading, h)
            if d < min_diff:
                min_diff = d
                best = name
        return best

    def _check_keypoints(self):
        """Check if the next unreached keypoint has been reached.

        Only checks the first unreached keypoint (in order).
        Requires BOTH coordinate proximity AND correct heading.
        Turn and forward keypoints both use INTERSECTION_VALID_RADIUS (10m).
        Destination keypoints use ORACLE_RADIUS (20m).
        """
        intersection_radius = self.intersection_valid_radius
        oracle_radius = self.oracle_radius

        # Find the next unreached keypoint
        for i, kp in enumerate(self.instruction_keypoints):
            if i in self.reached_instructions:
                continue

            dist = distance_meters(self.current_geocode, kp['geocode'])
            radius = oracle_radius if kp['is_destination'] else intersection_radius

            heading_ok = self._heading_diff(self.current_heading, kp['heading']) < self.HEADING_TOLERANCE
            if dist < radius and heading_ok:
                self.reached_instructions.add(i)
                return f"Reached checkpoint {kp['instr_idx']}."
            break  # only check the next unreached keypoint

        return ""

    def _get_observation_freeform(self):
        """Get text observation at current free-form position (for language_only mode)."""
        # Detect nearby landmarks from route landmarks
        landmark_obs = "No landmarks nearby"
        geocode_list = self.route_info['route_results']['geocode_list']
        landmark_strings = self.route_info['route_results'].get('landmark_list', [])

        for idx, lm_geocode in enumerate(geocode_list):
            dist = distance_meters(self.current_geocode, lm_geocode)
            if dist < self.oracle_radius:
                if idx < len(landmark_strings) and landmark_strings[idx] != "No landmark nearby":
                    landmark_obs = landmark_strings[idx]
                break

        # Rail metadata already records which positions are intersections.
        intersection_obs = ""
        if self.rail_idx < len(self.gt_rail_info):
            intersection_obs = self.gt_rail_info[self.rail_idx].get(
                "intersection_observation", ""
            )

        return f"{landmark_obs}; {intersection_obs}" if intersection_obs else landmark_obs

    def _move_on_rail(self, action):
        """Move to the next position on the rail (kept for oracle/debug mode)."""
        self.step_cnt += 1
        self.current_geocode = self.gt_rail_info[self.step_cnt]['geocode']
        self.traj_list.append(self.current_geocode)
        self.action_list.append(action)
        self.observation_list.append(self._get_observation_rail()['oracle_observation'])

    def _get_observation_rail(self):
        """Get observation at current rail position (kept for oracle/debug mode)."""
        intersection_obs = self.gt_rail_info[self.step_cnt]['intersection_observation']
        if self.gt_rail_info[self.step_cnt]['gt_action'] == "stop()":
            intersection_obs = ""
        landmark_obs = self.gt_rail_info[self.step_cnt]['observation']
        self.current_heading = self.gt_rail_info[self.step_cnt]['heading']

        obs_dict = {
            "oracle_observation": landmark_obs + '; ' + intersection_obs,
            "visual_observation": None if self.language_only else self._get_visual_observation()
        }
        return obs_dict

    def _get_visual_observation(self):
        """Fetch Street View images and compose into 2x2 grid as PIL Image.

        Layout: top-left=front, top-right=right, bottom-left=back, bottom-right=left.
        """
        heading_list = [
            self.current_heading % 360,         # front
            (self.current_heading + 90) % 360,  # right
            (self.current_heading + 180) % 360, # back
            (self.current_heading + 270) % 360, # left
        ]
        image_list = self.panorama_store.render_views(self.current_geocode, heading_list)

        line_width = 5
        canvas = Image.new('RGB', (self.resolution * 2 + line_width, self.resolution * 2 + line_width), (0, 0, 0))

        for i, image in enumerate(image_list):
            x = i % 2
            y = i // 2
            image = image.resize((self.resolution, self.resolution))
            pos_x = x * (self.resolution + line_width)
            pos_y = y * (self.resolution + line_width)
            canvas.paste(image, (pos_x, pos_y))

        return canvas

    def _render(self, init_obs=True):
        """Render the observation in HYGAE format."""
        img_placeholder = self.config.get("image_placeholder", "<image>")

        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )

        # Build multi_modal_data
        if not self.language_only:
            visual_obs = self._get_visual_observation()
            multi_modal_data = {img_placeholder: [visual_obs]}
            observation_text = img_placeholder
        else:
            multi_modal_data = {}
            observation_text = self._get_observation_freeform()

        # Build instruction progress checklist
        completed_instr_indices = set()
        for kp_idx in self.reached_instructions:
            completed_instr_indices.add(self.instruction_keypoints[kp_idx]['instr_idx'])

        instruction_progress = ""
        for i, instr in enumerate(self.list_instruction):
            marker = "[x]" if (i + 1) in completed_instr_indices else "[ ]"
            instruction_progress += f"{marker} {i + 1}. {instr}\n"
        instruction_progress = instruction_progress.rstrip("\n")

        # Build movement history
        if self.action_list:
            movement_history = ", ".join(self.action_list)
        else:
            movement_history = "(none)"

        # Compute facing direction (nearest compass name)
        facing_direction = self._heading_to_direction(self.current_heading)

        if init_obs:
            obs_str = init_observation_template(
                observation=observation_text,
                instruction_progress=instruction_progress,
                movement_history=movement_history,
                facing_direction=facing_direction,
            ) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=observation_text,
                reward=self.reward,
                done=self.info.get("metrics", {}).get("traj_metrics", {}).get("success", False),
                env_feedback=self.info.get("env_feedback", ""),
                instruction_progress=instruction_progress,
                movement_history=movement_history,
                facing_direction=facing_direction,
            ) + "\n" + format_prompt_text

        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }

    def system_prompt(self) -> str:
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True
        )
        return virl_system_prompt(
            format=self.config.prompt_format,
            absolute_action=self.absolute_action
        ) + '\n' + format_prompt_text

    def close(self):
        """Release cached references held by the environment."""
        self.panorama_store = None

    def _parse_instruction_and_rail(self):
        """Parse route milestone info to build instructions and ground truth rail.

        Adapted from NavigationEnvironment._parse_instruction_and_rail() in SFTvsRL.
        """
        intersection_dict_list = parse_navigation_string(self.route_info['milestone_info'])

        parsed_instruction = []
        intersection_actions = []
        relative_actions = []
        heading_list = [self.current_heading]
        landmark_list = []
        prev_heading = self.current_heading

        for idx, intersection_dict in enumerate(intersection_dict_list):
            landmark = intersection_dict.get('landmarks', "No landmark nearby")
            landmark_list.append(landmark)
            to_next_intersection_heading = intersection_dict.get('to_next_intersection_heading', None)
            next_heading = to_next_intersection_heading.split('(')[1].split(')')[0] if to_next_intersection_heading is not None else None
            next_heading_deg = int(to_next_intersection_heading.split('(')[0].strip()) if to_next_intersection_heading is not None else None
            heading_list.append(next_heading_deg)

            relative_heading_deg = next_heading_deg - prev_heading
            if relative_heading_deg < -45:
                relative_heading = 'left'
            elif relative_heading_deg > 45:
                relative_heading = 'right'
            elif -45 <= relative_heading_deg <= 0:
                relative_heading = 'slightly left'
            elif 0 <= relative_heading_deg <= 45:
                relative_heading = 'slightly right'

            prev_heading = next_heading_deg
            intersection_actions.append('turn_direction(' + next_heading + ')')
            relative_actions.append('turn_direction(' + relative_heading + ')')

            if idx == len(intersection_dict_list) - 1:
                parsed_instruction.append("Turn " + relative_heading + " to face " + next_heading + ".")
                parsed_instruction.append("Move forward until you reach destination where " + landmark + ".")
            else:
                first_msg = "First, turn" if idx == 0 else "Turn"
                if landmark == "No landmark nearby":
                    parsed_instruction.append(f"{first_msg} " + relative_heading + " to face " + next_heading + ".")
                    parsed_instruction.append("Move forward until you reach next intersection.")
                else:
                    parsed_instruction.append(f"{first_msg} " + relative_heading + " to face " + next_heading + ".")
                    parsed_instruction.append("Move forward until you reach next intersection where " + landmark + ".")

        str_instruction = ""
        for idx, instr in enumerate(parsed_instruction):
            str_instruction += f"{idx + 1}. {instr}\n"

        # Build ground truth rail
        self.gt_rail_info = []
        landmark_list.insert(0, "No landmarks nearby")
        intersection_actions.append("stop()")
        relative_actions.append("stop()")
        geocode_prev = self.current_geocode
        instruction_idx = 0

        for idx, geocode in enumerate([self.current_geocode] + self.route_info['route_results']['geocode_list']):
            distance = distance_meters(geocode_prev, geocode)

            if self.relocation:
                geocode, _ = self.panorama_store.relocate(tuple(geocode))

            if distance > 5:
                n_of_points = self.straight_line_length
                for i in range(n_of_points):
                    if random.random() < self.drop_rate:
                        continue
                    interpolated_geocode = self._interpolate_geocodes(geocode_prev, geocode, i + 1, n_of_points)
                    if self.relocation:
                        try:
                            relocated_geocode, _ = self.panorama_store.relocate(
                                tuple(interpolated_geocode)
                            )
                        except LookupError:
                            continue
                    else:
                        relocated_geocode = interpolated_geocode

                    if self.relocation:
                        if (distance_meters(geocode_prev, relocated_geocode) > 8 and
                                distance_meters(geocode, relocated_geocode) > 8):
                            self.gt_rail_info.append({
                                'geocode': relocated_geocode,
                                'heading': heading_list[idx],
                                'gt_action': "forward()",
                                'relative_action': "forward()",
                                'observation': "No landmarks nearby",
                                'intersection_observation': "",
                                'instruction': parsed_instruction[instruction_idx],
                                'instruction_idx': instruction_idx + 1,
                            })
                    else:
                        self.gt_rail_info.append({
                            'geocode': relocated_geocode,
                            'heading': heading_list[idx],
                            'gt_action': "forward()",
                            'observation': "No landmarks nearby",
                            'intersection_observation': "",
                            'instruction': parsed_instruction[instruction_idx],
                            'instruction_idx': instruction_idx + 1,
                        })
                instruction_idx += 1

            if intersection_actions[idx] == 'stop()':
                instruction_idx -= 1

            self.gt_rail_info.append({
                'geocode': geocode,
                'heading': heading_list[idx],
                'gt_action': intersection_actions[idx] if self.absolute_action else relative_actions[idx],
                'relative_action': relative_actions[idx],
                'observation': landmark_list[idx],
                'intersection_observation': "You observe an intersection" if (idx != 0) else "",
                'instruction': parsed_instruction[instruction_idx],
                'instruction_idx': instruction_idx + 1,
            })
            instruction_idx += 1
            geocode_prev = geocode

        return str_instruction, parsed_instruction

    @staticmethod
    def _interpolate_geocodes(geocode_prev, geocode_next, i, n):
        """Linearly interpolate between two geocode positions."""
        return (
            geocode_prev[0] * (1 - i / n) + geocode_next[0] * (i / n),
            geocode_prev[1] * (1 - i / n) + geocode_next[1] * (i / n)
        )
