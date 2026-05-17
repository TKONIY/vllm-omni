#!/usr/bin/env python3
"""Profile MoE routing from Tencent's official HunyuanImage-3.0 code path."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
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
    parser.add_argument("--official-repo", default="artifacts/HunyuanImage-3.0-official")
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--prompt", default=LONG_PROMPT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--moe-impl", default="flashinfer", choices=["eager", "flashinfer"])
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--use-system-prompt", default="en_unified")
    parser.add_argument("--bot-task", default="think")
    parser.add_argument(
        "--deterministic-ar",
        action="store_true",
        help="Use greedy AR text generation to align with vLLM temperature=0 runs.",
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


_TRACE_CONTEXT: dict[str, Any] = {
    "stage": "ar",
    "default_modality": "ar_text",
    "labels": None,
}


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


def _set_index_labels(
    labels: torch.Tensor,
    indices: torch.Tensor | None,
    value: int,
) -> None:
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
    from transformers.cache_utils import StaticLayer

    from vllm_omni.model_executor.models.hunyuan_image3 import moe_route_trace

    original_lazy_initialization = StaticLayer.lazy_initialization

    def compat_lazy_initialization(self, key_states, value_states=None):
        if value_states is None:
            value_states = key_states
        return original_lazy_initialization(self, key_states, value_states)

    original_gate_forward = official_modeling.HunyuanTopKGate.forward
    original_model_forward = official_modeling.HunyuanImage3ForCausalMM.forward
    original_prepare_inputs = official_modeling.HunyuanImage3ForCausalMM.prepare_inputs_for_generation
    original_update_kwargs = official_modeling.HunyuanImage3ForCausalMM._update_model_kwargs_for_generation

    def traced_gate_forward(self, hidden_states, topk_impl="default"):
        output = original_gate_forward(self, hidden_states, topk_impl=topk_impl)
        topk_indices = None
        if topk_impl == "easy" and isinstance(output, tuple) and len(output) >= 2:
            topk_indices = output[1]
        if topk_indices is not None:
            labels = _TRACE_CONTEXT.get("labels")
            if labels is not None:
                labels = labels.to(topk_indices.device)
            moe_route_trace.record_routes(
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
            stage = "official_dit"
            default_modality = "dit_unknown"
        else:
            labels = None
            stage = "official_ar"
            default_modality = "ar_text"
        with trace_context(stage, default_modality, labels):
            return original_model_forward(self, *args, **kwargs)

    def compat_prepare_inputs(self, *args, **kwargs):
        model_inputs = original_prepare_inputs(self, *args, **kwargs)
        model_inputs.setdefault("use_cache", kwargs.get("use_cache", True))
        return model_inputs

    def compat_update_kwargs(self, outputs, model_kwargs, *args, **kwargs):
        updated = original_update_kwargs(self, outputs, model_kwargs, *args, **kwargs)
        updated.setdefault("use_cache", model_kwargs.get("use_cache", True))
        return updated

    traced_gate_forward.__signature__ = inspect.signature(original_gate_forward)  # type: ignore[attr-defined]
    traced_model_forward.__signature__ = inspect.signature(original_model_forward)  # type: ignore[attr-defined]
    compat_prepare_inputs.__signature__ = inspect.signature(original_prepare_inputs)  # type: ignore[attr-defined]
    compat_update_kwargs.__signature__ = inspect.signature(original_update_kwargs)  # type: ignore[attr-defined]
    compat_lazy_initialization.__signature__ = inspect.signature(original_lazy_initialization)  # type: ignore[attr-defined]
    StaticLayer.lazy_initialization = compat_lazy_initialization
    official_modeling.HunyuanTopKGate.forward = traced_gate_forward
    official_modeling.HunyuanImage3ForCausalMM.forward = traced_model_forward
    official_modeling.HunyuanImage3ForCausalMM.prepare_inputs_for_generation = compat_prepare_inputs
    official_modeling.HunyuanImage3ForCausalMM._update_model_kwargs_for_generation = compat_update_kwargs


def main() -> None:
    args = parse_args()
    root = repo_root()
    official_repo = (root / args.official_repo).resolve()
    if not official_repo.exists():
        raise SystemExit(f"Official repo not found: {official_repo}")

    if args.output_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = root / "artifacts" / "uad_official_hunyuan_image3_routing" / stamp
    else:
        output_root = Path(args.output_root).resolve()
    trace_dir = output_root / "trace"
    image_dir = output_root / "images"
    trace_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HUNYUAN_MOE_ROUTE_TRACE_DIR"] = str(trace_dir)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(official_repo))

    install_official_route_trace()

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

    generate_kwargs = {}
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
    image_path = image_dir / "official_output_0.png"
    samples[0].save(image_path)
    (output_root / "cot_text.txt").write_text(str(cot_text), encoding="utf-8")

    from vllm_omni.model_executor.models.hunyuan_image3 import moe_route_trace

    moe_route_trace.flush()

    image_url = f"images/{image_path.name}" if image_path.exists() else ""
    html_path = output_root / "hunyuan_official_routing.html"
    json_path = output_root / "route_report.json"
    height, width = parse_image_size(args.image_size)
    import subprocess

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
            f"official:{args.model}",
            "--stage-config",
            str(official_repo),
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
    print(f"HTML: {html_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Image: {image_path}", flush=True)


if __name__ == "__main__":
    main()
