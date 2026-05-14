#!/usr/bin/env python3
"""Run EP=2 HunyuanImage3 FusedMoE backend sweeps.

Each backend/mode runs in a fresh torchrun process because vLLM kernel
selection, distributed state, and CUDA workspaces are process-global.
Failures are recorded and the sweep continues.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BACKENDS = (
    "auto,"
    "flashinfer_trtllm,"
    "flashinfer_cutlass,"
    "triton,"
    "deep_gemm,"
    "deep_gemm_mega_moe,"
    "cutlass,"
    "flashinfer_cutedsl,"
    "marlin,"
    "aiter,"
    "emulation"
)

DEFAULT_MODES = "topk8_balanced,topk1_single"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="tencent/HunyuanImage-3.0-Instruct")
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--tokens", default="16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-iters", type=int, default=20)
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def tail_text(text: str, limit: int = 5000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "ep2_backend_sweep_summary.json"

    script = Path(__file__).resolve().with_name("profile_hunyuan_moe_ep2_case.py")
    modes = split_csv(args.modes)
    backends = split_csv(args.backends)
    records: list[dict[str, Any]] = []

    for mode in modes:
        for backend in backends:
            stem = f"{mode}_{backend}"
            output_json = output_dir / f"{stem}.json"
            log_path = output_dir / f"{stem}.log"

            if args.skip_existing and output_json.exists():
                records.append(
                    {
                        "mode": mode,
                        "backend": backend,
                        "status": "ok",
                        "returncode": 0,
                        "elapsed_s": 0.0,
                        "output_json": str(output_json),
                        "log": str(log_path),
                        "skipped_existing": True,
                    }
                )
                continue

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(args.nproc_per_node),
                str(script),
                "--model-path",
                args.model_path,
                "--layer-id",
                str(args.layer_id),
                "--mode",
                mode,
                "--expert-id",
                str(args.expert_id),
                "--tokens",
                args.tokens,
                "--output-json",
                str(output_json),
                "--seed",
                str(args.seed),
                "--warmup",
                str(args.warmup),
                "--event-iters",
                str(args.event_iters),
                "--moe-backend",
                backend,
                "--all2all-backend",
                args.all2all_backend,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

            print(f"=== mode={mode} backend={backend} ===", flush=True)
            start = time.time()
            completed = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            elapsed = time.time() - start
            log_path.write_text(completed.stdout + "\n" + completed.stderr)
            status = "ok" if completed.returncode == 0 and output_json.exists() else "failed"
            record = {
                "mode": mode,
                "backend": backend,
                "status": status,
                "returncode": completed.returncode,
                "elapsed_s": elapsed,
                "output_json": str(output_json) if output_json.exists() else "",
                "log": str(log_path),
                "error_tail": "" if status == "ok" else tail_text(completed.stdout + "\n" + completed.stderr),
            }
            records.append(record)
            print(
                f"mode={mode} backend={backend} status={status} "
                f"rc={completed.returncode} elapsed={elapsed:.1f}s",
                flush=True,
            )

            write_json(
                summary_path,
                {
                    "title": "HunyuanImage3 Layer-15 EP=2 FusedMoE Backend Sweep",
                    "model_path": args.model_path,
                    "layer_id": args.layer_id,
                    "expert_id": args.expert_id,
                    "tokens": split_csv(args.tokens),
                    "modes": modes,
                    "backends": backends,
                    "cuda_visible_devices": args.cuda_visible_devices,
                    "nproc_per_node": args.nproc_per_node,
                    "all2all_backend": args.all2all_backend,
                    "records": records,
                },
            )

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
