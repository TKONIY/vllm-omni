#!/usr/bin/env python3
"""Profile official HunyuanImage3 routing in a clean upstream environment.

This runner intentionally does not patch Transformers compatibility APIs.  It
only instruments the official HunyuanImage3 gate/model forward functions to
record MoE top-k routes with the same JSON schema used by the vLLM-side routing
reports.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

LONG_PROMPT = (
    "生成一张复杂的咖啡馆图片：清晨的玻璃屋顶咖啡馆里有多层空间，前景是一杯拉花清晰的拿铁，"
    "杯壁上有水汽和反光；桌面散落着手写菜单、咖啡豆、铜色量勺和半打开的旧书。中景有一台正在"
    "工作的复古意式咖啡机，蒸汽穿过斜射阳光形成可见光束；背景能看到雨后的街道、霓虹倒影、"
    "植物墙、木质楼梯和几位神态不同的顾客。画面需要写实摄影质感，细节丰富，构图有纵深，"
    "暖色室内光和冷色窗外光形成对比。"
)

LABEL_TO_MODALITY = {
    0: "dit_text",
    1: "dit_image",
    2: "dit_timestep",
    3: "dit_cond_image",
    4: "dit_other",
}

_TRACE_CONTEXT: dict[str, Any] = {
    "stage": "official_ar_clean",
    "default_modality": "ar_text",
    "labels": None,
}
_TRACE: dict[str, Any] = {
    "schema_version": 1,
    "created_at": time.time(),
    "metadata": {},
    "stages": {},
}
_HAS_DATA = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path() -> str:
    hub_dir = Path.home() / ".cache/huggingface/hub/models--tencent--HunyuanImage-3.0-Instruct/snapshots"
    snapshots = sorted(hub_dir.glob("*"))
    if snapshots:
        return str(snapshots[-1])
    return "tencent/HunyuanImage-3.0-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", default="../HunyuanImage-3.0-official-clean")
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--prompt", default=LONG_PROMPT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--moe-impl", default="flashinfer", choices=["eager", "flashinfer"])
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--use-system-prompt", default="en_unified")
    parser.add_argument("--bot-task", default="think")
    parser.add_argument(
        "--deterministic-ar",
        action="store_true",
        help="Use greedy AR text generation for vLLM-aligned comparisons.",
    )
    parser.add_argument("--rank-policy", choices=["rank0", "all"], default="rank0")
    return parser.parse_args()


def parse_image_size(image_size: str) -> tuple[int, int]:
    if "x" in image_size:
        h_text, w_text = image_size.lower().split("x", 1)
    elif ":" in image_size:
        h_text, w_text = image_size.split(":", 1)
    else:
        value = int(image_size)
        return value, value
    return int(h_text), int(w_text)


@contextmanager
def trace_context(stage: str, default_modality: str, labels: torch.Tensor | None) -> Iterator[None]:
    previous = dict(_TRACE_CONTEXT)
    _TRACE_CONTEXT["stage"] = stage
    _TRACE_CONTEXT["default_modality"] = default_modality
    _TRACE_CONTEXT["labels"] = labels
    try:
        yield
    finally:
        _TRACE_CONTEXT.update(previous)


def _rank_metadata() -> dict[str, Any]:
    data: dict[str, Any] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        if torch.cuda.is_available():
            data["cuda_current_device"] = torch.accelerator.current_device_index()
    except Exception:
        pass
    return data


def _ensure_stage(stage: str, num_experts: int, top_k: int) -> dict[str, Any]:
    stages = _TRACE.setdefault("stages", {})
    entry = stages.setdefault(
        stage,
        {
            "num_experts": num_experts,
            "top_k": top_k,
            "num_calls": 0,
            "global": {},
            "by_layer": {},
        },
    )
    entry["num_experts"] = max(int(entry.get("num_experts", 0)), int(num_experts))
    entry["top_k"] = int(top_k)
    return entry


def _empty_modality_entry(num_experts: int) -> dict[str, Any]:
    return {
        "route_counts": [0 for _ in range(num_experts)],
        "token_positions": 0,
        "topk_assignments": 0,
        "calls": 0,
    }


def _accumulate(bucket: dict[str, Any], modality: str, counts: list[int], token_positions: int) -> None:
    entry = bucket.setdefault(modality, _empty_modality_entry(len(counts)))
    if len(entry["route_counts"]) < len(counts):
        entry["route_counts"].extend([0] * (len(counts) - len(entry["route_counts"])))
    for idx, value in enumerate(counts):
        entry["route_counts"][idx] += int(value)
    entry["token_positions"] += int(token_positions)
    entry["topk_assignments"] += int(sum(counts))
    entry["calls"] += 1


def _counts(topk_indices: torch.Tensor, num_experts: int) -> list[int]:
    flat = topk_indices.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts)
    return [int(v) for v in counts.detach().cpu().tolist()]


def record_routes(
    *,
    stage: str,
    layer_id: int,
    topk_indices: torch.Tensor,
    num_experts: int,
    default_modality: str,
    labels: torch.Tensor | None,
) -> None:
    global _HAS_DATA
    with torch.no_grad():
        topk_indices = topk_indices.detach()
        top_k = int(topk_indices.shape[-1]) if topk_indices.ndim > 1 else 1
        num_tokens = int(topk_indices.reshape(-1, top_k).shape[0])
        stage_entry = _ensure_stage(stage, num_experts, top_k)
        stage_entry["num_calls"] += 1
        layer_key = str(int(layer_id))
        layer_entry = stage_entry["by_layer"].setdefault(layer_key, {})

        if labels is None:
            counts = _counts(topk_indices, num_experts)
            _accumulate(stage_entry["global"], default_modality, counts, num_tokens)
            _accumulate(layer_entry, default_modality, counts, num_tokens)
            _HAS_DATA = True
            return

        active_labels = labels.detach().reshape(-1).to(topk_indices.device)
        if active_labels.numel() != num_tokens:
            counts = _counts(topk_indices, num_experts)
            _accumulate(stage_entry["global"], default_modality, counts, num_tokens)
            _accumulate(layer_entry, default_modality, counts, num_tokens)
            stage_entry.setdefault("warnings", []).append(
                {
                    "layer_id": int(layer_id),
                    "message": (
                        f"label/token mismatch: labels={active_labels.numel()} "
                        f"tokens={num_tokens}; used {default_modality}"
                    ),
                }
            )
            _HAS_DATA = True
            return

        flat_topk = topk_indices.reshape(num_tokens, top_k)
        for label_id, modality in LABEL_TO_MODALITY.items():
            mask = active_labels == label_id
            token_positions = int(mask.sum().item())
            if token_positions == 0:
                continue
            counts = _counts(flat_topk[mask], num_experts)
            _accumulate(stage_entry["global"], modality, counts, token_positions)
            _accumulate(layer_entry, modality, counts, token_positions)
            _HAS_DATA = True


def flush_trace(trace_dir: Path) -> Path | None:
    if not _HAS_DATA:
        return None
    trace_dir.mkdir(parents=True, exist_ok=True)
    _TRACE["metadata"] = _rank_metadata()
    _TRACE["finished_at"] = time.time()
    stages = "-".join(sorted(_TRACE.get("stages", {}).keys())) or "empty"
    rank = _TRACE["metadata"].get("cuda_current_device", "na")
    path = trace_dir / f"hunyuan_moe_route_trace_{stages}_rank{rank}_pid{os.getpid()}.json"
    path.write_text(json.dumps(_TRACE, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _set_index_labels(labels: torch.Tensor, indices: torch.Tensor | None, value: int) -> None:
    if indices is None:
        return
    if indices.ndim == 1:
        labels[:, indices.to(labels.device).long()] = value
    else:
        row = torch.arange(indices.shape[0], device=labels.device).unsqueeze(-1)
        labels[row, indices.to(labels.device).long()] = value


def _set_mask_labels(labels: torch.Tensor, mask: torch.Tensor | None, value: int) -> None:
    if mask is None:
        return
    labels[mask.to(labels.device).bool()] = value


def _image_token_count(model: Any, images: Any) -> tuple[int, torch.device]:
    if isinstance(images, torch.Tensor):
        patch = int(getattr(model.config, "patch_size", 1) or 1)
        return int((images.shape[-2] // patch) * (images.shape[-1] // patch)), images.device
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, torch.Tensor):
            patch = int(getattr(model.config, "patch_size", 1) or 1)
            return int((first.shape[-2] // patch) * (first.shape[-1] // patch)), first.device
    raise ValueError(f"Unsupported image batch type for trace labels: {type(images)!r}")


def build_dit_labels(model: Any, kwargs: dict[str, Any]) -> torch.Tensor | None:
    input_ids = kwargs.get("input_ids")
    if input_ids is not None:
        labels = torch.zeros(input_ids.shape, dtype=torch.int64, device=input_ids.device)
        _set_mask_labels(labels, kwargs.get("image_mask"), 1)
        _set_index_labels(labels, kwargs.get("timesteps_index"), 2)
        _set_index_labels(labels, kwargs.get("guidance_index"), 4)
        _set_index_labels(labels, kwargs.get("timesteps_r_index"), 4)
        _set_mask_labels(labels, kwargs.get("cond_vae_image_mask"), 3)
        _set_mask_labels(labels, kwargs.get("cond_vit_image_mask"), 3)
        _set_index_labels(labels, kwargs.get("cond_timesteps_index"), 2)
        return labels.reshape(-1)

    images = kwargs.get("images")
    if images is None:
        return None
    image_tokens, device = _image_token_count(model, images)
    parts = [torch.full((1,), 2, dtype=torch.int64, device=device)]
    if kwargs.get("guidance") is not None:
        parts.append(torch.full((1,), 4, dtype=torch.int64, device=device))
    if kwargs.get("timesteps_r") is not None:
        parts.append(torch.full((1,), 4, dtype=torch.int64, device=device))
    parts.append(torch.full((image_tokens,), 1, dtype=torch.int64, device=device))
    one = torch.cat(parts, dim=0)
    batch = int(images.shape[0]) if isinstance(images, torch.Tensor) else len(images)
    return one.unsqueeze(0).repeat(batch, 1).reshape(-1)


def install_official_route_trace() -> None:
    from hunyuan_image_3 import modeling_hunyuan_image_3 as official_modeling

    original_gate_forward = official_modeling.HunyuanTopKGate.forward
    original_model_forward = official_modeling.HunyuanImage3ForCausalMM.forward

    def traced_gate_forward(self, hidden_states, topk_impl="default"):
        output = original_gate_forward(self, hidden_states, topk_impl=topk_impl)
        topk_indices = None
        if topk_impl == "easy" and isinstance(output, tuple) and len(output) >= 2:
            topk_indices = output[1]
        if topk_indices is not None:
            labels = _TRACE_CONTEXT.get("labels")
            if labels is not None:
                labels = labels.to(topk_indices.device)
            record_routes(
                stage=str(_TRACE_CONTEXT["stage"]),
                layer_id=int(self.layer_idx if self.layer_idx is not None else -1),
                topk_indices=topk_indices,
                num_experts=int(self.wg.out_features),
                default_modality=str(_TRACE_CONTEXT["default_modality"]),
                labels=labels,
            )
        return output

    def traced_model_forward(self, *args, **kwargs):
        mode = kwargs.get("mode") or "gen_text"
        if mode == "gen_image":
            labels = build_dit_labels(self, kwargs)
            stage = "official_dit_clean"
            default_modality = "dit_unknown"
        else:
            labels = None
            stage = "official_ar_clean"
            default_modality = "ar_text"
        with trace_context(stage, default_modality, labels):
            return original_model_forward(self, *args, **kwargs)

    traced_gate_forward.__signature__ = inspect.signature(original_gate_forward)  # type: ignore[attr-defined]
    traced_model_forward.__signature__ = inspect.signature(original_model_forward)  # type: ignore[attr-defined]
    official_modeling.HunyuanTopKGate.forward = traced_gate_forward
    official_modeling.HunyuanImage3ForCausalMM.forward = traced_model_forward


def main() -> None:
    args = parse_args()
    root = repo_root()
    official_repo = (root / args.official_repo).resolve()
    if not official_repo.exists():
        raise SystemExit(f"Official repo not found: {official_repo}")

    if args.output_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = root / "artifacts" / "uad_official_hunyuan_image3_routing_clean" / stamp
    else:
        output_root = Path(args.output_root).resolve()
    trace_dir = output_root / "trace"
    image_dir = output_root / "images"
    trace_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(official_repo))

    install_official_route_trace()

    import transformers
    from hunyuan_image_3 import HunyuanImage3ForCausalMM

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
    )

    model = HunyuanImage3ForCausalMM.from_pretrained(args.model, **kwargs)
    model.load_tokenizer(args.model)
    model.generation_config.diff_infer_steps = args.steps
    model.generation_config.diff_guidance_scale = args.guidance_scale
    if args.deterministic_ar:
        model.generation_config.do_sample = False

    generate_kwargs: dict[str, Any] = {}
    if args.deterministic_ar:
        generate_kwargs["do_sample"] = False

    cot_text, samples = model.generate_image(
        prompt=args.prompt,
        seed=args.seed,
        image_size=args.image_size,
        use_system_prompt=args.use_system_prompt,
        bot_task=args.bot_task,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        verbose=1,
        **generate_kwargs,
    )
    image_path = image_dir / "official_clean_output_0.png"
    samples[0].save(image_path)
    (output_root / "cot_text.txt").write_text(str(cot_text), encoding="utf-8")

    trace_path = flush_trace(trace_dir)
    image_url = f"images/{image_path.name}" if image_path.exists() else ""
    html_path = output_root / "hunyuan_official_clean_routing.html"
    json_path = output_root / "route_report.json"
    height, width = parse_image_size(args.image_size)

    subprocess.run(
        [
            sys.executable,
            str(root / "docs/uad/script/build_hunyuan_real_request_routing_html.py"),
            "--trace-dir",
            str(trace_dir),
            "--output-html",
            str(html_path),
            "--output-json",
            str(json_path),
            "--prompt",
            args.prompt,
            "--model",
            f"official-clean:{args.model}",
            "--stage-config",
            f"{official_repo} | transformers={transformers.__version__} | moe_impl={args.moe_impl}",
            "--height",
            str(height),
            "--width",
            str(width),
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--seed",
            str(args.seed),
            "--image-url",
            image_url,
            "--rank-policy",
            args.rank_policy,
        ],
        cwd=root,
        check=True,
    )

    print(f"Output root: {output_root}", flush=True)
    print(f"Trace: {trace_path}", flush=True)
    print(f"HTML: {html_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Image: {image_path}", flush=True)


if __name__ == "__main__":
    main()
