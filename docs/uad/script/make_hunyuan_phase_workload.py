#!/usr/bin/env python3
"""Generate JSONL workloads for HunyuanImage3 phase-imbalance experiments.

The generated records are intentionally plain. They can be replayed by
run_hunyuan_phase_load.py against the OpenAI-compatible
/v1/images/generations endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SHORT_PROMPTS = [
    "A complex latte art photograph on a marble cafe table.",
    "A futuristic espresso machine in a small Tokyo cafe.",
    "A ceramic cup of cappuccino with warm morning light.",
    "A detailed still life of coffee beans, milk foam, and glassware.",
    "A photorealistic iced coffee beside a laptop and handwritten notes.",
    "A cinematic close-up of a barista pouring milk into espresso.",
    "A cozy cafe counter with pastries, plants, and brass lighting.",
    "A macro photo of crema on freshly pulled espresso.",
]

LONG_PROMPTS = [
    (
        "Create a highly detailed image of a busy specialty coffee workshop with several workstations, "
        "transparent glass drippers, copper kettles, labeled jars of beans from different origins, "
        "handwritten tasting notes, a wall-mounted roasting schedule, soft daylight through tall windows, "
        "and subtle reflections on the polished concrete floor. The composition should include many small "
        "objects, realistic shadows, natural color variation, and a documentary photography style."
    ),
    (
        "Generate an intricate scene inside a two-level cafe library. The foreground contains a pour-over "
        "setup with a scale, timer, blooming coffee bed, folded linen towel, and a half-open notebook. "
        "The background has shelves of books, people quietly reading, pendant lights, steam drifting upward, "
        "and a rainy city street visible through the window. Use a realistic lens perspective and make the "
        "small details readable without becoming cluttered."
    ),
    (
        "Design a complex commercial product photograph for a premium coffee subscription box. Show the box, "
        "multiple bags of beans, tasting cards, a grinder, a kettle, a ceramic mug, scattered roasted beans, "
        "and packaging textures. Use studio lighting, accurate materials, soft but visible shadows, sharp "
        "focus across the important objects, and a balanced composition suitable for a magazine spread."
    ),
    (
        "Illustrate a crowded morning cafe from a slightly elevated angle with a long bar, glass pastry case, "
        "espresso machines, reflected light on metal surfaces, customers ordering, a barista workflow, menu "
        "boards, hanging plants, and cups at different preparation stages. Keep the scene coherent, realistic, "
        "and rich in detail, with a warm but not oversaturated palette."
    ),
]


DEFAULTS = {
    "dit_heavy": {"width": 1024, "height": 1024, "steps": 50},
    "ar_heavy": {"width": 512, "height": 512, "steps": 8},
    "bursty_mix": {"width": 1024, "height": 1024, "steps": 50},
}


def parse_size(size: str | None) -> tuple[int | None, int | None]:
    if size is None:
        return None, None
    if "x" not in size.lower():
        raise ValueError(f"size must look like WIDTHxHEIGHT, got {size!r}")
    width, height = size.lower().split("x", 1)
    return int(width), int(height)


def load_prompt_file(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            prompts.append(line)
            continue
        if isinstance(payload, str):
            prompts.append(payload)
        elif isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            prompts.append(payload["prompt"])
        else:
            raise ValueError(f"{path}:{line_no} must be text, a JSON string, or a JSON object with prompt")
    if not prompts:
        raise ValueError(f"{path} did not contain any prompts")
    return prompts


def choose_kind(profile: str, index: int, total: int, rng: random.Random) -> str:
    if profile in {"dit_heavy", "ar_heavy"}:
        return profile
    first_cut = max(total // 3, 1)
    second_cut = max((2 * total) // 3, first_cut + 1)
    if index < first_cut:
        return "dit_heavy" if rng.random() < 0.8 else "ar_heavy"
    if index < second_cut:
        return "ar_heavy" if rng.random() < 0.8 else "dit_heavy"
    return "dit_heavy" if rng.random() < 0.5 else "ar_heavy"


def make_record(
    *,
    index: int,
    total: int,
    profile: str,
    prompt_prefix: str,
    prompt_pool: list[str] | None,
    rng: random.Random,
    width_override: int | None,
    height_override: int | None,
    steps_override: int | None,
    guidance_scale: float,
    seed_base: int,
) -> dict[str, Any]:
    kind = choose_kind(profile, index, total, rng)
    defaults = DEFAULTS[kind]
    width = width_override if width_override is not None else defaults["width"]
    height = height_override if height_override is not None else defaults["height"]
    steps = steps_override if steps_override is not None else defaults["steps"]
    builtins = LONG_PROMPTS if kind == "ar_heavy" else SHORT_PROMPTS
    prompt_source = prompt_pool if prompt_pool is not None else builtins
    prompt = prompt_source[index % len(prompt_source)]
    if prompt_prefix:
        prompt = f"{prompt_prefix} {prompt}"
    request_id = f"{profile}-{index:06d}"
    return {
        "request_id": request_id,
        "profile": profile,
        "prompt_kind": kind,
        "prompt": prompt,
        "width": width,
        "height": height,
        "size": f"{width}x{height}",
        "steps": steps,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed_base + index,
        "metadata": {
            "generator": Path(__file__).name,
            "profile_index": index,
            "profile_total": total,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for workload mixture choices.")
    parser.add_argument("--seed-base", type=int, default=42, help="Base image seed written into request records.")
    parser.add_argument("--size", default=None, help="Override image size for all records, for example 1024x1024.")
    parser.add_argument("--steps", type=int, default=None, help="Override diffusion steps for all records.")
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Optional text/JSONL prompt source.")
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")

    width, height = parse_size(args.size)
    prompt_pool = load_prompt_file(args.prompt_file) if args.prompt_file is not None else None
    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for index in range(args.num_requests):
            record = make_record(
                index=index,
                total=args.num_requests,
                profile=args.profile,
                prompt_prefix=args.prompt_prefix,
                prompt_pool=prompt_pool,
                rng=rng,
                width_override=width,
                height_override=height,
                steps_override=args.steps,
                guidance_scale=args.guidance_scale,
                seed_base=args.seed_base,
            )
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Wrote {args.num_requests} requests to {output}")


if __name__ == "__main__":
    main()
