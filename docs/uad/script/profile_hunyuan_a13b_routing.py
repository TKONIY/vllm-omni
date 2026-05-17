#!/usr/bin/env python3
"""Run one Hunyuan-A13B request and trace MoE routing for prefill/decode."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "tencent/Hunyuan-A13B-Instruct"
DEFAULT_PROMPT = (
    "生成一张复杂的咖啡馆图片：清晨的玻璃屋顶咖啡馆里有多层空间，前景是一杯拉花清晰的拿铁，"
    "杯壁上有水汽和反光；桌面散落着手写菜单、咖啡豆、铜色量勺和半打开的旧书。中景有一台"
    "正在工作的复古意式咖啡机，蒸汽穿过斜射阳光形成可见光束；背景能看到雨后的街道、霓虹"
    "倒影、植物墙、木质楼梯和几位神态不同的顾客。画面需要写实摄影质感，细节丰富，构图有"
    "纵深，暖色室内光和冷色窗外光形成对比。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--rank-policy", choices=["rank0", "all"], default="rank0")
    parser.add_argument("--publish-dir", default=None)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_python(root: Path) -> str:
    venv_python = root / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def write_sitecustomize(path: Path) -> None:
    path.write_text(
        r'''
from __future__ import annotations

import os


def _patch_hunyuan_a13b_moe_trace() -> None:
    if not os.environ.get("HUNYUAN_MOE_ROUTE_TRACE_DIR"):
        return
    try:
        import torch
        import vllm.model_executor.models.hunyuan_v1 as hy
        from vllm_omni.model_executor.models.hunyuan_image3.moe_route_trace import (
            record_routes,
        )
    except Exception:
        return

    cls = hy.HunYuanSparseMoeBlock
    if getattr(cls, "_uad_route_trace_patched", False):
        return

    orig_init = cls.__init__

    def traced_init(self, *args, **kwargs):
        layer_id = kwargs.get("layer_id", -1)
        if len(args) >= 3:
            layer_id = args[2]
        orig_init(self, *args, **kwargs)
        self._uad_route_trace_layer_id = int(layer_id)

    def traced_forward(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat_hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(flat_hidden_states)
        with torch.no_grad():
            capture_file = os.environ.get("HUNYUAN_A13B_ROUTE_CAPTURE_FILE")
            if capture_file and os.path.exists(capture_file):
                top_k = int(getattr(self.experts, "top_k", 1))
                num_experts = int(
                    getattr(
                        self,
                        "n_routed_experts",
                        getattr(self.experts, "logical_num_experts", router_logits.shape[-1]),
                    )
                )
                # Single-request vLLM execution: prefill forwards many tokens,
                # decode forwards one new token per step.
                modality = "a13b_prefill" if flat_hidden_states.shape[0] > 1 else "a13b_decode"
                _, topk_indices = torch.topk(router_logits.float(), top_k, dim=-1)
                record_routes(
                    stage="a13b",
                    layer_id=int(getattr(self, "_uad_route_trace_layer_id", -1)),
                    topk_indices=topk_indices,
                    num_experts=num_experts,
                    default_modality=modality,
                )

        final_hidden_states = self.experts(
            hidden_states=flat_hidden_states,
            router_logits=router_logits,
        )
        return final_hidden_states.view(orig_shape)

    cls.__init__ = traced_init
    cls.forward = traced_forward
    cls._uad_route_trace_patched = True


_patch_hunyuan_a13b_moe_trace()
'''.lstrip(),
        encoding="utf-8",
    )


def render_child_script(path: Path, args: argparse.Namespace, output_root: Path, trace_dir: Path) -> None:
    path.write_text(
        f"""
from __future__ import annotations

import json
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = {args.model!r}
    prompt = {args.prompt!r}
    output_root = Path({str(output_root)!r})

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    messages = [{{"role": "user", "content": prompt}}]
    try:
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        rendered_prompt = prompt

    llm = LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size={args.tp},
        distributed_executor_backend="mp",
        enforce_eager=True,
        max_model_len={args.max_model_len},
        max_num_batched_tokens={args.max_num_batched_tokens},
        gpu_memory_utilization={args.gpu_memory_utilization},
    )
    Path({str(output_root / "capture_enabled")!r}).write_text("1", encoding="utf-8")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens={args.max_tokens},
    )
    outputs = llm.generate([rendered_prompt], sampling_params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    (output_root / "generation_output.txt").write_text(text, encoding="utf-8")
    (output_root / "run_meta.json").write_text(
        json.dumps(
            {{
                "model": model,
                "prompt": prompt,
                "rendered_prompt": rendered_prompt,
                "rendered_prompt_tokens": len(tokenizer.encode(rendered_prompt)),
                "max_tokens": {args.max_tokens},
                "tp": {args.tp},
                "devices": {args.devices!r},
                "trace_dir": {str(trace_dir)!r},
            }},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )


def copy_publish(output_root: Path, publish_dir: Path) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_root / "hunyuan_a13b_routing.html", publish_dir / "index.html")
    shutil.copy2(output_root / "route_report.json", publish_dir / "route_report.json")


def main() -> None:
    args = parse_args()
    root = repo_root()
    if args.output_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = root / "artifacts" / "uad_a13b_routing" / stamp
    else:
        output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trace_dir = output_root / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = output_root / "patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    write_sitecustomize(patch_dir / "sitecustomize.py")
    child_script = output_root / "run_hunyuan_a13b_child.py"
    render_child_script(child_script, args, output_root, trace_dir)

    python_exe = repo_python(root)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["HUNYUAN_MOE_ROUTE_TRACE_DIR"] = str(trace_dir)
    env["HUNYUAN_A13B_ROUTE_CAPTURE_FILE"] = str(output_root / "capture_enabled")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = (
        str(patch_dir)
        + os.pathsep
        + str(root)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )

    import subprocess

    log_path = output_root / "generation.log"
    cmd = [python_exe, str(child_script)]
    print("Running:", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace")[-6000:]
        raise SystemExit(f"Hunyuan-A13B request failed with code {proc.returncode}. Log tail:\\n{tail}")

    html_path = output_root / "hunyuan_a13b_routing.html"
    json_path = output_root / "route_report.json"
    build_cmd = [
        python_exe,
        "docs/uad/script/build_hunyuan_real_request_routing_html.py",
        "--trace-dir",
        str(trace_dir),
        "--output-html",
        str(html_path),
        "--output-json",
        str(json_path),
        "--prompt",
        args.prompt,
        "--model",
        args.model,
        "--stage-config",
        f"vLLM LLM(tp={args.tp}, devices={args.devices})",
        "--height",
        "0",
        "--width",
        "0",
        "--steps",
        "0",
        "--guidance-scale",
        "0",
        "--seed",
        "0",
        "--rank-policy",
        args.rank_policy,
    ]
    subprocess.run(build_cmd, cwd=root, check=True)

    if args.publish_dir:
        copy_publish(output_root, Path(args.publish_dir).resolve())
        print(f"Published page files to {args.publish_dir}", flush=True)

    print(f"Output root: {output_root}", flush=True)
    print(f"HTML: {html_path}", flush=True)
    print(f"Trace dir: {trace_dir}", flush=True)
    print(f"Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
