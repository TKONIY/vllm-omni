#!/usr/bin/env python3
"""Open-loop load generator for HunyuanImage3 online phase experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} must contain a JSON object")
        records.append(payload)
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def json_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_request_payload(
    record: dict[str, Any],
    response_format: str,
    extra_payload: dict[str, Any],
) -> dict[str, Any]:
    width = int(record.get("width", 1024))
    height = int(record.get("height", 1024))
    payload: dict[str, Any] = {
        "prompt": record["prompt"],
        "response_format": response_format,
        "size": record.get("size", f"{width}x{height}"),
        "num_inference_steps": int(record.get("num_inference_steps", record.get("steps", 50))),
        "guidance_scale": float(record.get("guidance_scale", 0.0)),
    }
    if "seed" in record:
        payload["seed"] = int(record["seed"])
    payload.update(extra_payload)
    return payload


def compact_response_json(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"response_type": type(payload).__name__}
    summary: dict[str, Any] = {}
    for key in ("id", "created", "model", "usage", "metrics", "timings", "stage_durations"):
        if key in payload:
            summary[key] = payload[key]
    data = payload.get("data")
    if isinstance(data, list):
        summary["data_count"] = len(data)
        data_keys: list[str] = []
        for item in data:
            if isinstance(item, dict):
                data_keys = sorted({*data_keys, *item.keys()})
        summary["data_keys"] = [key for key in data_keys if key not in {"b64_json", "url"}]
    return summary


def find_stage_durations(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    direct = payload.get("stage_durations")
    if isinstance(direct, dict):
        return direct
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("stage_durations"), dict):
        return metrics["stage_durations"]
    timings = payload.get("timings")
    if isinstance(timings, dict):
        return timings
    return {}


def post_json(url: str, payload: dict[str, Any], timeout_s: float, api_key: str | None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read()
            http_status = int(response.status)
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        http_status = int(exc.code)
        response_headers = dict(exc.headers.items()) if exc.headers else {}
    completed = time.time()

    parsed: Any
    try:
        parsed = json.loads(response_body.decode("utf-8"))
    except Exception:
        parsed = None
    return {
        "http_status": http_status,
        "ok": 200 <= http_status < 300,
        "response_bytes": len(response_body),
        "request_wall_s": completed - started,
        "response_headers": response_headers,
        "response_metrics": compact_response_json(parsed),
        "stage_durations": find_stage_durations(parsed),
        "error_body": None if 200 <= http_status < 300 else response_body[:4096].decode("utf-8", errors="replace"),
    }


def make_schedule(
    records: list[dict[str, Any]],
    *,
    rate: float,
    duration_s: float | None,
    max_requests: int | None,
    seed: int,
) -> list[tuple[int, float, dict[str, Any]]]:
    if rate <= 0:
        raise ValueError("--rate must be positive")
    if duration_s is None and max_requests is None:
        max_requests = len(records)
    rng = random.Random(seed)
    schedule: list[tuple[int, float, dict[str, Any]]] = []
    offset_s = 0.0
    index = 0
    while True:
        if duration_s is not None and offset_s > duration_s:
            break
        if max_requests is not None and index >= max_requests:
            break
        schedule.append((index, offset_s, records[index % len(records)]))
        index += 1
        offset_s += rng.expovariate(rate)
    return schedule


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


async def sample_nvidia_smi(path: Path, interval_s: float, stop: asyncio.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    query = "index,utilization.gpu,utilization.memory,memory.used,power.draw"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    with path.open("a", encoding="utf-8") as f:
        while not stop.is_set():
            sample_ts = time.time()
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    f.write(json_dumps({"sample_ts": sample_ts, "ok": False, "error": proc.stderr.strip()}) + "\n")
                else:
                    for line in proc.stdout.splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) != 5:
                            continue
                        f.write(
                            json_dumps(
                                {
                                    "sample_ts": sample_ts,
                                    "ok": True,
                                    "gpu_index": int(parts[0]),
                                    "util_gpu_pct": parse_float(parts[1]),
                                    "util_mem_pct": parse_float(parts[2]),
                                    "memory_used_mib": parse_float(parts[3]),
                                    "power_w": parse_float(parts[4]),
                                }
                            )
                            + "\n"
                        )
                f.flush()
            except Exception as exc:
                f.write(json_dumps({"sample_ts": sample_ts, "ok": False, "error": repr(exc)}) + "\n")
                f.flush()
            await asyncio.sleep(interval_s)


def fetch_metrics_text(url: str, keep_regex: re.Pattern[str] | None, max_chars: int) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode("utf-8", errors="replace")
        status = int(response.status)
    if keep_regex is not None:
        lines = [line for line in text.splitlines() if keep_regex.search(line)]
        text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars]
    return status, text


async def sample_metrics(
    *,
    url: str,
    path: Path,
    interval_s: float,
    keep_regex: re.Pattern[str] | None,
    max_chars: int,
    stop: asyncio.Event,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        while not stop.is_set():
            sample_ts = time.time()
            try:
                status, text = await asyncio.to_thread(fetch_metrics_text, url, keep_regex, max_chars)
                f.write(json_dumps({"sample_ts": sample_ts, "ok": True, "http_status": status, "text": text}) + "\n")
            except Exception as exc:
                f.write(json_dumps({"sample_ts": sample_ts, "ok": False, "error": repr(exc)}) + "\n")
            f.flush()
            await asyncio.sleep(interval_s)


async def send_one(
    *,
    sequence_id: int,
    scheduled_offset_s: float,
    record: dict[str, Any],
    start_mono: float,
    start_epoch: float,
    endpoint_url: str,
    response_format: str,
    extra_payload: dict[str, Any],
    timeout_s: float,
    api_key: str | None,
    semaphore: asyncio.Semaphore,
    output_file: Any,
    output_lock: asyncio.Lock,
    progress_every: int,
    counter: dict[str, int],
) -> None:
    scheduled_epoch = start_epoch + scheduled_offset_s
    await asyncio.sleep(max(0.0, start_mono + scheduled_offset_s - time.monotonic()))
    payload = build_request_payload(record, response_format, extra_payload)
    ready_at = time.time()
    result: dict[str, Any] = {
        "request_id": record.get("request_id", f"request-{sequence_id:06d}"),
        "sequence_id": sequence_id,
        "profile": record.get("profile"),
        "prompt_kind": record.get("prompt_kind"),
        "width": record.get("width"),
        "height": record.get("height"),
        "steps": record.get("steps", record.get("num_inference_steps")),
        "seed": record.get("seed"),
        "scheduled_offset_s": scheduled_offset_s,
        "scheduled_at": scheduled_epoch,
        "client_ready_at": ready_at,
    }
    try:
        async with semaphore:
            sent_at = time.time()
            result["sent_at"] = sent_at
            result["client_queue_delay_s"] = max(0.0, sent_at - scheduled_epoch)
            response = await asyncio.to_thread(post_json, endpoint_url, payload, timeout_s, api_key)
        completed_at = time.time()
        result.update(response)
        result["completed_at"] = completed_at
        result["latency_s"] = completed_at - sent_at
        result["status"] = "ok" if response["ok"] else "error"
        if not response["ok"]:
            result["error"] = response.get("error_body")
    except TimeoutError as exc:
        completed_at = time.time()
        sent_at = float(result.get("sent_at", ready_at))
        result.update(
            {
                "completed_at": completed_at,
                "latency_s": completed_at - sent_at,
                "status": "timeout",
                "error": repr(exc),
            }
        )
    except Exception as exc:
        completed_at = time.time()
        sent_at = float(result.get("sent_at", ready_at))
        result.update(
            {
                "completed_at": completed_at,
                "latency_s": completed_at - sent_at,
                "status": "error",
                "error": repr(exc),
            }
        )

    async with output_lock:
        output_file.write(json_dumps(result) + "\n")
        output_file.flush()
        counter["done"] += 1
        done = counter["done"]
        if progress_every > 0 and done % progress_every == 0:
            print(f"completed {done} requests")


async def run(args: argparse.Namespace) -> None:
    workload = load_jsonl(Path(args.workload))
    schedule = make_schedule(
        workload,
        rate=args.rate,
        duration_s=args.duration_s,
        max_requests=args.max_requests,
        seed=args.seed,
    )
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_payload = json.loads(args.extra_payload_json) if args.extra_payload_json else {}
    if not isinstance(extra_payload, dict):
        raise ValueError("--extra-payload-json must decode to a JSON object")
    endpoint_url = args.base_url.rstrip("/") + args.endpoint

    if args.dry_run:
        with output.open("w", encoding="utf-8") as f:
            for sequence_id, scheduled_offset_s, record in schedule:
                payload = build_request_payload(record, args.response_format, extra_payload)
                f.write(
                    json_dumps(
                        {
                            "request_id": record.get("request_id", f"request-{sequence_id:06d}"),
                            "sequence_id": sequence_id,
                            "profile": record.get("profile"),
                            "prompt_kind": record.get("prompt_kind"),
                            "scheduled_offset_s": scheduled_offset_s,
                            "status": "dry_run",
                            "payload_keys": sorted(payload),
                            "size": payload.get("size"),
                            "num_inference_steps": payload.get("num_inference_steps"),
                        }
                    )
                    + "\n"
                )
        print(f"Dry-run wrote {len(schedule)} planned requests to {output}")
        return

    stop = asyncio.Event()
    sampler_tasks: list[asyncio.Task[None]] = []
    if args.nvidia_smi_jsonl:
        sampler_tasks.append(
            asyncio.create_task(sample_nvidia_smi(Path(args.nvidia_smi_jsonl), args.nvidia_smi_interval_s, stop))
        )
    if args.metrics_jsonl:
        metrics_url = args.metrics_url or args.base_url.rstrip("/") + "/metrics"
        keep_regex = re.compile(args.metrics_filter_regex) if args.metrics_filter_regex else None
        sampler_tasks.append(
            asyncio.create_task(
                sample_metrics(
                    url=metrics_url,
                    path=Path(args.metrics_jsonl),
                    interval_s=args.metrics_interval_s,
                    keep_regex=keep_regex,
                    max_chars=args.metrics_max_chars,
                    stop=stop,
                )
            )
        )

    start_mono = time.monotonic()
    start_epoch = time.time()
    semaphore = asyncio.Semaphore(args.client_max_concurrency)
    output_lock = asyncio.Lock()
    counter = {"done": 0}
    with output.open("w", encoding="utf-8") as f:
        tasks = [
            asyncio.create_task(
                send_one(
                    sequence_id=sequence_id,
                    scheduled_offset_s=scheduled_offset_s,
                    record=record,
                    start_mono=start_mono,
                    start_epoch=start_epoch,
                    endpoint_url=endpoint_url,
                    response_format=args.response_format,
                    extra_payload=extra_payload,
                    timeout_s=args.timeout_s,
                    api_key=args.api_key,
                    semaphore=semaphore,
                    output_file=f,
                    output_lock=output_lock,
                    progress_every=args.progress_every,
                    counter=counter,
                )
            )
            for sequence_id, scheduled_offset_s, record in schedule
        ]
        print(f"scheduled {len(tasks)} requests at {args.rate} req/s against {endpoint_url}")
        try:
            await asyncio.gather(*tasks)
        finally:
            stop.set()
            if sampler_tasks:
                await asyncio.gather(*sampler_tasks, return_exceptions=True)

    print(f"Wrote request results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--endpoint", default="/v1/images/generations")
    parser.add_argument("--rate", type=float, required=True, help="Open-loop offered rate in requests/second.")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--client-max-concurrency", type=int, default=1024)
    parser.add_argument("--response-format", default="b64_json")
    parser.add_argument("--extra-payload-json", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--nvidia-smi-jsonl", default=None)
    parser.add_argument("--nvidia-smi-interval-s", type=float, default=1.0)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--metrics-url", default=None)
    parser.add_argument(
        "--metrics-filter-regex",
        default="vllm|omni|diffusion|scheduler|queue|running|waiting|stage",
    )
    parser.add_argument("--metrics-interval-s", type=float, default=1.0)
    parser.add_argument("--metrics-max-chars", type=int, default=200_000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
