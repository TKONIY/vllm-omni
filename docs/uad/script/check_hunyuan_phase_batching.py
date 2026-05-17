#!/usr/bin/env python3
"""Preflight checks for HunyuanImage3 phase-internal continuous batching.

This script is intentionally static: it reads source/config files and does not
load model weights. It is used as Gate 0 before running expensive online
serving experiments.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_rev(root: Path, ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", ref],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_text(root: Path, relpath: str) -> str:
    return (root / relpath).read_text(encoding="utf-8")


def load_yaml(root: Path, relpath: str) -> dict[str, Any]:
    with (root / relpath).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{relpath} did not parse to a mapping")
    return data


def get_stage(config: dict[str, Any], stage_id: int) -> dict[str, Any]:
    for stage in config.get("stages", []):
        if int(stage.get("stage_id", -1)) == stage_id:
            return stage
    raise ValueError(f"stage_id={stage_id} not found")


def stage_max_num_seqs(stage: dict[str, Any]) -> int | None:
    value = stage.get("max_num_seqs")
    if value is None:
        engine_args = stage.get("engine_args")
        if isinstance(engine_args, dict):
            value = engine_args.get("max_num_seqs")
    return None if value is None else int(value)


def check_ar_sampler(root: Path) -> dict[str, Any]:
    source = read_text(root, "vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py")
    sample_match = re.search(r"\n    def sample\([\s\S]*?\n    def _get_forced_token", source)
    sample_source = sample_match.group(0) if sample_match else ""
    batch_loop = "for req_idx in range(logits.shape[0])" in sample_source
    legacy_single_assert = "logits.shape[0] == 1" in sample_source or "batch_size == 1" in sample_source
    sampler_call = "self._sampler(logits=logits, sampling_metadata=sampling_metadata)" in sample_source
    return {
        "source": "vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py",
        "sample_method_found": bool(sample_match),
        "per_request_logits_loop": batch_loop,
        "legacy_single_batch_assert": legacy_single_assert,
        "uses_vllm_sampler": sampler_call,
        "phase_internal_batching_supported_by_model_code": bool(
            sample_match and batch_loop and not legacy_single_assert and sampler_call
        ),
    }


def check_dit_step_mode(root: Path) -> dict[str, Any]:
    pipeline_source = read_text(root, "vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py")
    engine_source = read_text(root, "vllm_omni/diffusion/diffusion_engine.py")
    required_methods = ["prepare_encode", "denoise_step", "step_scheduler", "post_decode"]
    found_methods = {name: bool(re.search(rf"\n    def {name}\(", pipeline_source)) for name in required_methods}
    supports_flag = bool(re.search(r"supports_step_execution\s*:\s*ClassVar\[bool\]\s*=\s*True", pipeline_source))
    has_downgrade_guard = "Non-stepwise-execution does not support max-num-seqs" in engine_source
    return {
        "pipeline_source": "vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py",
        "engine_source": "vllm_omni/diffusion/diffusion_engine.py",
        "step_methods": found_methods,
        "supports_step_execution_flag": supports_flag,
        "diffusion_engine_downgrades_non_stepwise_max_num_seqs": has_downgrade_guard,
        "phase_internal_batching_supported_by_model_code": bool(
            supports_flag and all(found_methods.values())
        ),
    }


def check_deploy_config(root: Path, deploy_config: str) -> dict[str, Any]:
    config = load_yaml(root, deploy_config)
    ar_stage = get_stage(config, 0)
    dit_stage = get_stage(config, 1)
    edges = config.get("edges", [])
    edge_0_1 = next(
        (
            edge
            for edge in edges
            if int(edge.get("from", -1)) == 0 and int(edge.get("to", -1)) == 1
        ),
        {},
    )
    return {
        "deploy_config": deploy_config,
        "ar_max_num_seqs": stage_max_num_seqs(ar_stage),
        "dit_max_num_seqs": stage_max_num_seqs(dit_stage),
        "edge_0_to_1_max_inflight": edge_0_1.get("max_inflight"),
        "edge_0_to_1_window_size": edge_0_1.get("window_size"),
    }


def build_report(root: Path, deploy_config: str) -> dict[str, Any]:
    ar = check_ar_sampler(root)
    dit = check_dit_step_mode(root)
    deploy = check_deploy_config(root, deploy_config)
    ar_config_allows_batch = deploy["ar_max_num_seqs"] is not None and deploy["ar_max_num_seqs"] > 1
    dit_config_allows_batch = deploy["dit_max_num_seqs"] is not None and deploy["dit_max_num_seqs"] > 1
    edge_allows_multiple_inflight = deploy["edge_0_to_1_max_inflight"] in (None, -1) or (
        isinstance(deploy["edge_0_to_1_max_inflight"], int) and deploy["edge_0_to_1_max_inflight"] > 1
    )
    return {
        "repo": str(root),
        "head": git_rev(root, "HEAD"),
        "origin_main": git_rev(root, "origin/main"),
        "ar": ar,
        "dit": dit,
        "deploy": deploy,
        "gate": {
            "ar_model_can_batch": ar["phase_internal_batching_supported_by_model_code"],
            "ar_default_config_allows_batch": ar_config_allows_batch,
            "dit_model_can_stepwise_batch": dit["phase_internal_batching_supported_by_model_code"],
            "dit_default_config_allows_batch": dit_config_allows_batch,
            "edge_default_allows_multiple_inflight": edge_allows_multiple_inflight,
            "full_phase_internal_continuous_batching_ready": bool(
                ar["phase_internal_batching_supported_by_model_code"]
                and ar_config_allows_batch
                and dit["phase_internal_batching_supported_by_model_code"]
                and dit_config_allows_batch
                and edge_allows_multiple_inflight
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/hunyuan_image3.yaml")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-blocker", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    report = build_report(root, args.deploy_config)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    if args.fail_on_blocker and not report["gate"]["full_phase_internal_continuous_batching_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
