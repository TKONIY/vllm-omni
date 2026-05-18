#!/usr/bin/env python3
"""Create a HunyuanImage3 deploy YAML for the version-A imbalance experiment.

The default bundled YAML serializes both the AR stage and the AR->DiT edge.
This helper raises AR/edge concurrency while leaving DiT in request mode. The
current DiffusionEngine still runs non-stepwise DiT with one request at a time,
so this config measures the real separated baseline rather than a hypothetical
stepwise DiT implementation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal script environments.
    yaml = None


def get_stage(config: dict[str, Any], stage_id: int) -> dict[str, Any]:
    for stage in config.get("stages", []):
        if int(stage.get("stage_id", -1)) == stage_id:
            return stage
    raise ValueError(f"stage_id={stage_id} not found")


def rewrite_known_hunyuan_yaml(
    text: str,
    *,
    ar_max_num_seqs: int,
    ar_max_num_batched_tokens: int,
    dit_max_num_seqs: int,
    edge_max_inflight: int,
) -> str:
    """Fallback editor for the bundled deploy YAML when PyYAML is absent."""
    lines = text.splitlines()
    current_stage: int | None = None
    in_edges = False
    current_edge_from: int | None = None
    current_edge_to: int | None = None
    seen = {
        "ar_max_num_seqs": False,
        "ar_max_num_batched_tokens": False,
        "dit_max_num_seqs": False,
        "edge_max_inflight": False,
    }
    output: list[str] = []

    for line in lines:
        stage_match = re.match(r"^  - stage_id:\s*(\d+)\s*$", line)
        edge_from_match = re.match(r"^  - from:\s*(\d+)\s*$", line)
        edge_to_match = re.match(r"^    to:\s*(\d+)\s*$", line)
        if line.startswith("edges:"):
            in_edges = True
            current_stage = None
        elif re.match(r"^[a-zA-Z_]+:", line) and not line.startswith("edges:"):
            in_edges = False
        if stage_match and not in_edges:
            current_stage = int(stage_match.group(1))
        if in_edges and edge_from_match:
            current_edge_from = int(edge_from_match.group(1))
            current_edge_to = None
        if in_edges and edge_to_match:
            current_edge_to = int(edge_to_match.group(1))

        new_line = line
        if current_stage == 0 and re.match(r"^    max_num_seqs:", line):
            new_line = f"    max_num_seqs: {ar_max_num_seqs}"
            seen["ar_max_num_seqs"] = True
        elif current_stage == 0 and re.match(r"^    max_num_batched_tokens:", line):
            new_line = f"    max_num_batched_tokens: {ar_max_num_batched_tokens}"
            seen["ar_max_num_batched_tokens"] = True
        elif current_stage == 1 and re.match(r"^    max_num_seqs:", line):
            new_line = f"    max_num_seqs: {dit_max_num_seqs}"
            seen["dit_max_num_seqs"] = True
        elif (
            in_edges
            and current_edge_from == 0
            and current_edge_to == 1
            and re.match(r"^    max_inflight:", line)
        ):
            new_line = f"    max_inflight: {edge_max_inflight}"
            seen["edge_max_inflight"] = True
        output.append(new_line)

    missing = [name for name, was_seen in seen.items() if not was_seen]
    if missing:
        raise ValueError(f"fallback YAML rewrite could not find fields: {missing}")
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="vllm_omni/deploy/hunyuan_image3.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ar-max-num-seqs", type=int, default=8)
    parser.add_argument("--ar-max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--dit-max-num-seqs", type=int, default=1)
    parser.add_argument("--edge-max-inflight", type=int, default=64)
    args = parser.parse_args()

    base_text = Path(args.base).read_text(encoding="utf-8")
    if yaml is None:
        body = rewrite_known_hunyuan_yaml(
            base_text,
            ar_max_num_seqs=args.ar_max_num_seqs,
            ar_max_num_batched_tokens=args.ar_max_num_batched_tokens,
            dit_max_num_seqs=args.dit_max_num_seqs,
            edge_max_inflight=args.edge_max_inflight,
        )
    else:
        config = yaml.safe_load(base_text)
        if not isinstance(config, dict):
            raise ValueError(f"{args.base} did not parse to a mapping")

        ar_stage = get_stage(config, 0)
        dit_stage = get_stage(config, 1)
        ar_stage["max_num_seqs"] = args.ar_max_num_seqs
        ar_stage["max_num_batched_tokens"] = args.ar_max_num_batched_tokens
        dit_stage["max_num_seqs"] = args.dit_max_num_seqs

        for edge in config.get("edges", []):
            if int(edge.get("from", -1)) == 0 and int(edge.get("to", -1)) == 1:
                edge["max_inflight"] = args.edge_max_inflight
        body = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by docs/uad/script/make_hunyuan_phase_deploy_config.py.\n"
        "# Version A: AR concurrency and AR->DiT inflight are opened; DiT remains request-mode single request.\n"
    )
    output.write_text(header + body, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
