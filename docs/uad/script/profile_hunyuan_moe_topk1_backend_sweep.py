#!/usr/bin/env python3
"""Sweep vLLM MoE backends for the TopK=1 same-expert Hunyuan control.

Each backend is run in a separate Python process because vLLM backend
selection and workspace setup are process-global enough that reusing a single
process can hide backend-specific initialization failures.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_BACKENDS = (
    "auto,"
    "flashinfer_cutlass,"
    "flashinfer_trtllm,"
    "triton,"
    "deep_gemm,"
    "deep_gemm_mega_moe,"
    "cutlass,"
    "flashinfer_cutedsl,"
    "marlin,"
    "aiter,"
    "emulation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="tencent/HunyuanImage-3.0-Instruct")
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--tokens", default="16384,32768,65536")
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-iters", type=int, default=20)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = ["backend", "status", "tokens", "component", "error", "returncode"]
    for key in preferred:
        seen.add(key)
        fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truncate_error(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    rows: list[dict[str, Any]] = []
    backends = split_csv(args.backends)

    script = Path(__file__).resolve().with_name("profile_hunyuan_moe_topk1_single_expert.py")
    python = sys.executable

    for backend in backends:
        with tempfile.TemporaryDirectory(prefix=f"uad_topk1_{backend}_") as tmp_dir:
            tmp_csv = Path(tmp_dir) / "profile.csv"
            cmd = [
                python,
                str(script),
                "--model-path",
                args.model_path,
                "--layer-id",
                str(args.layer_id),
                "--expert-id",
                str(args.expert_id),
                "--tokens",
                args.tokens,
                "--output",
                str(tmp_csv),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--warmup",
                str(args.warmup),
                "--event-iters",
                str(args.event_iters),
                "--moe-backend",
                backend,
            ]
            print(f"=== backend={backend} ===", flush=True)
            completed = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                error = truncate_error(completed.stdout + "\n" + completed.stderr)
                rows.append(
                    {
                        "backend": backend,
                        "status": "failed",
                        "tokens": "",
                        "component": "",
                        "error": error,
                        "returncode": completed.returncode,
                    }
                )
                print(f"backend={backend} failed rc={completed.returncode}", flush=True)
                continue

            backend_rows = read_csv(tmp_csv)
            if not backend_rows:
                rows.append(
                    {
                        "backend": backend,
                        "status": "failed",
                        "tokens": "",
                        "component": "",
                        "error": "backend completed but wrote no rows",
                        "returncode": completed.returncode,
                    }
                )
                print(f"backend={backend} wrote no rows", flush=True)
                continue

            for row in backend_rows:
                row["backend"] = backend
                row["status"] = "ok"
                row["component"] = row.get("category", "")
                row["error"] = ""
                row["returncode"] = completed.returncode
                rows.append(row)
            print(f"backend={backend} ok rows={len(backend_rows)}", flush=True)

    write_rows(output, rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
