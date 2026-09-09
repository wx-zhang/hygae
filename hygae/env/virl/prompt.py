FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """e.g.
Round 1:
[image]
Navigation Instruction:
[ ] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: (none)
<think>I see a street view with buildings on both sides. The instructions say: turn south, move forward, until the next intersection. I should first turn south and then move forward and see if there is intersection.</think>
<answer>turn_direction(south), forward()</answer>
Round 2:
Env_feedback: Executed 2 action(s). Reached checkpoint 1.
[image]
Navigation Instruction:
[x] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: turn_direction(south), forward()
<think>I am still on the road. From the instruction, i have not reach the next intersection, i need to keep forward.</think>
<answer>forward()</answer>
"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """e.g.
Round 1:
[image]
Navigation Instruction:
[ ] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: (none)
<answer>turn_direction(south), forward()</answer>
Round 2:
Env_feedback: Executed 2 action(s). Reached checkpoint 1.
[image]
Navigation Instruction:
[x] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: turn_direction(south), forward()
<answer>forward()</answer>
"""
    },
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should be described in detail about what you see in the street view images.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """e.g.
Round 1:
[image]
Navigation Instruction:
[ ] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: (none)
<think><observation>I see a street view with buildings on both sides.</observation><reasoning>The instructions say: turn south, move forward, until the next intersection.</reasoning></think>
<answer>turn_direction(south), forward()</answer>
Round 2:
Env_feedback: Executed 2 action(s). Reached checkpoint 1.
[image]
Navigation Instruction:
[x] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: turn_direction(south), forward()
<think><observation>I am still on the road</observation><reasoning>From the instruction, i have not reach the next intersection, i need to keep forward</reasoning></think>
<answer>forward()</answer>
"""
    },
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.\nThe prediction should describe what you expect to see after your action is executed.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """e.g.
Round 1:
[image]
Navigation Instruction:
[ ] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: (none)
<think><reasoning>The instructions say: turn south, move forward, until the next intersection.</reasoning><prediction>I should first turn south and then move forward and see if there is intersection.</prediction></think>
<answer>turn_direction(south), forward()</answer>
Round 2:
Env_feedback: Executed 2 action(s). Reached checkpoint 1.
[image]
Navigation Instruction:
[x] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: turn_direction(south), forward()
<think><reasoning>From the instruction, i have not reach the next intersection, i need to keep forward</reasoning><prediction>I need to forward untill the next intersection</prediction></think>
<answer>forward()</answer>
"""
    },
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with your observation, reasoning, and prediction of next state, then your answer.\nBoth the observation and prediction should describe what you see or expect to see in the street view images.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """e.g.
Round 1:
[image]
Navigation Instruction:
[ ] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: (none)
<think><observation>I see a street view with buildings on both sides.</observation><reasoning>The instructions say: turn south, move forward, until the next intersection.</reasoning><prediction>I should first turn south and then move forward and see if there is intersection.</prediction></think>
<answer>turn_direction(south), forward()</answer>
Round 2:
Env_feedback: Executed 2 action(s). Reached checkpoint 1.
[image]
Navigation Instruction:
[x] 1. First, turn south to face south.
[ ] 2. Move forward until you reach next intersection.
[ ] 3. Turn right to face north.
[ ] 4. Move forward until you reach destination.
Movement history: turn_direction(south), forward()
<think><observation>I am still on the road</observation><reasoning>From the instruction, i have not reach the next interaction, i need to keep forward</reasoning><prediction>>I need to forward untill the next interaction</prediction></think>
<answer>forward()</answer>
"""
    }
}


def system_prompt(**kwargs):
    absolute_action = kwargs.get("absolute_action", True)

    if absolute_action:
        action_space_desc = """Actions you can take: forward(), turn_direction(x), stop()
forward(): Move forward one step along the road.
turn_direction(x): Adjust your direction towards x. x can be one of 8 compass directions: north, northeast, east, southeast, south, southwest, west, northwest.
stop(): Indicate that you have reached the destination and the navigation is finished."""
    else:
        action_space_desc = """Actions you can take: forward(), turn_direction(x), stop()
forward(): Move forward one step along the road.
turn_direction(x): Adjust your direction towards x. x can be one of: left, right, slightly left, slightly right.
stop(): Indicate that you have reached the destination and the navigation is finished."""

    base_prompt_text = f"""You are a navigation agent following instructions on real-world streets using Google Street View images.
You observe a 2x2 grid of street view images from your current position. Top-left: front view, top-right: right view, bottom-left: back view, bottom-right: left view.
Your current facing direction will be indicated with each observation.
{action_space_desc}
The navigation instruction will be provided with each observation. Look at the street view images carefully and follow the instructions to navigate to the destination.
Hints:
1. Pay attention to landmarks and intersections mentioned in the instructions.
2. When you see an intersection, check your instructions to decide whether to turn or continue forward.
3. If the instruction requires turn, always turn no matter where you are facing. 
4. If the instruction requires move forward, you may need to do multiple move forward actions until the you reach the required position. 
"""
    return base_prompt_text


def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction_progress = kwargs.get("instruction_progress", "")
    movement_history = kwargs.get("movement_history", "(none)")
    facing_direction = kwargs.get("facing_direction", "unknown")

    text = f"""[Initial Observation]:
{observation}
You are currently facing {facing_direction}.
Navigation Instruction:
{instruction_progress}
Movement history: {movement_history}
Decide your next action."""
    return text


def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction_progress = kwargs.get("instruction_progress", "")
    movement_history = kwargs.get("movement_history", "(none)")
    facing_direction = kwargs.get("facing_direction", "unknown")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")

    text = f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
reward: {reward}
done: {done}
After that, the observation is:
{observation}
You are currently facing {facing_direction}.
Navigation Instruction:
{instruction_progress}
Movement history: {movement_history}
Decide your next action."""
    return text


def format_prompt_generator(format_type):
    """Generate a prompt function for the specified VIRL navigation format type."""
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]

        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
{config["description"]}
Your response should be in the format of:
{config["format"]}"""

        if add_example:
            return base_prompt + '\n' + config["example"]

        return base_prompt

    return prompt_function


format_prompt = {
    ft: format_prompt_generator(ft)
    for ft in FORMAT_CONFIGS
}
