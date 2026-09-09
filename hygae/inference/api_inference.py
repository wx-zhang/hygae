"""Evaluate an OpenAI-compatible vision-language model in a local environment."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from hygae.env import REGISTERED_ENV, register_env


def _image_part(image):
    """Convert one PIL image to an OpenAI image content part."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def _observation_content(observation, leading_images=()):
    """Interleave an observation's text and images for Chat Completions."""

    text = observation["obs_str"]
    images = list(leading_images)
    images.extend(observation.get("multi_modal_data", {}).get("<image>", []))
    if not images:
        return text

    parts = [_image_part(image) for image in leading_images]
    remaining_images = images[len(leading_images) :]
    text_parts = text.split("<image>")
    for index, text_part in enumerate(text_parts):
        if text_part:
            parts.append({"type": "text", "text": text_part})
        if index < len(text_parts) - 1 and remaining_images:
            parts.append(_image_part(remaining_images.pop(0)))
    parts.extend(_image_part(image) for image in remaining_images)
    return parts


def _jsonable(value):
    """Convert environment metrics to values accepted by json.dumps."""

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_episode(client, model, record, max_turns, max_tokens, temperature):
    """Run one parquet record without an environment or inference server."""

    extra_info = record["extra_info"]
    env_name = extra_info["env_name"]
    seed = int(extra_info["seed"])
    register_env(env_name)
    entry = REGISTERED_ENV[env_name]
    env = entry["env_cls"](entry["config_cls"](**extra_info["env_config"]))

    try:
        observation, info = env.reset(seed=seed)
        system_prompt = env.system_prompt()

        # Some visual-prompt environments place example-image placeholders in the
        # system prompt. Chat Completions only accepts images in user messages, so
        # attach those example images to the first user turn instead.
        initial_images = observation.get("multi_modal_data", {}).get("<image>", [])
        example_count = system_prompt.count("<image>")
        example_images = initial_images[:example_count]
        if example_count:
            system_prompt = system_prompt.replace(
                "<image>", "[example image attached to the first user message]"
            )
            observation = dict(observation)
            observation["multi_modal_data"] = {
                "<image>": initial_images[example_count:]
            }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _observation_content(observation, example_images),
            },
        ]
        transcript = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation["obs_str"]},
        ]
        total_reward = 0.0
        done = False

        for _ in range(max_turns):
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response})
            transcript.append({"role": "assistant", "content": response})

            observation, reward, done, info = env.step(response)
            total_reward += float(reward)
            if done:
                break

            messages.append(
                {"role": "user", "content": _observation_content(observation)}
            )
            transcript.append({"role": "user", "content": observation["obs_str"]})

        total_reward += float(env.compute_reward())
        return {
            "env_name": env_name,
            "seed": seed,
            "score": total_reward,
            "done": done,
            "turns": sum(item["role"] == "assistant" for item in transcript),
            "metrics": _jsonable(info.get("metrics", {})),
            "transcript": transcript,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible API directly in HYGAE environments."
    )
    parser.add_argument("--model", required=True, help="API model name")
    parser.add_argument("--data", default="data/sokoban/test.parquet")
    parser.add_argument("--output", default="outputs/api_inference.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    if args.limit is not None:
        frame = frame.head(args.limit)

    client_options = {}
    if args.base_url:
        client_options["base_url"] = args.base_url
    client = OpenAI(**client_options)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, record in frame.iterrows():
            result = run_episode(
                client,
                args.model,
                record,
                args.max_turns,
                args.max_tokens,
                args.temperature,
            )
            result["index"] = int(index)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            scores.append(result["score"])
            print(
                f"[{len(scores)}/{len(frame)}] seed={result['seed']} "
                f"score={result['score']:.4f} done={result['done']}"
            )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Saved {len(scores)} episodes to {output_path}; mean score={mean_score:.4f}")


if __name__ == "__main__":
    main()
