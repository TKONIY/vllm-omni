#!/usr/bin/env python3
"""Build an HTML report for HunyuanImage3 phase-imbalance load runs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def summarize_run(path: Path, records: list[dict[str, Any]], slo_values: list[float]) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    completed = [
        record
        for record in records
        if record.get("completed_at") is not None and record.get("sent_at") is not None
    ]
    latencies = [float(record["latency_s"]) for record in ok if record.get("latency_s") is not None]
    sent_times = [float(record["sent_at"]) for record in completed]
    completed_times = [float(record["completed_at"]) for record in completed]
    span_s = max(completed_times) - min(sent_times) if sent_times and completed_times else None
    throughput = len(ok) / span_s if span_s and span_s > 0 else None
    profile_counts = Counter(str(record.get("prompt_kind") or record.get("profile") or "unknown") for record in records)
    slo_goodput = {
        str(slo): (sum(1 for value in latencies if value <= slo) / span_s if span_s and span_s > 0 else None)
        for slo in slo_values
    }
    return {
        "name": run_name(path),
        "path": str(path),
        "total": len(records),
        "ok": len(ok),
        "error": sum(1 for record in records if record.get("status") == "error"),
        "timeout": sum(1 for record in records if record.get("status") == "timeout"),
        "dry_run": sum(1 for record in records if record.get("status") == "dry_run"),
        "success_rate": len(ok) / len(records) if records else None,
        "span_s": span_s,
        "throughput_req_s": throughput,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p90_s": percentile(latencies, 90),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "slo_goodput_req_s": slo_goodput,
        "profile_counts": dict(sorted(profile_counts.items())),
    }


def latency_series(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    points = [
        (float(record["completed_at"]), float(record["latency_s"]))
        for record in records
        if record.get("status") == "ok"
        and record.get("completed_at") is not None
        and record.get("latency_s") is not None
    ]
    if not points:
        return {"name": name, "x": [], "y": []}
    t0 = min(t for t, _ in points)
    return {"name": name, "x": [t - t0 for t, _ in points], "y": [latency for _, latency in points]}


def inflight_series(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[tuple[float, int]] = []
    for record in records:
        if record.get("sent_at") is None or record.get("completed_at") is None:
            continue
        events.append((float(record["sent_at"]), 1))
        events.append((float(record["completed_at"]), -1))
    if not events:
        return {"name": name, "x": [], "y": []}
    events.sort()
    t0 = events[0][0]
    current = 0
    xs: list[float] = []
    ys: list[int] = []
    for event_time, delta in events:
        current += delta
        xs.append(event_time - t0)
        ys.append(current)
    return {"name": name, "x": xs, "y": ys}


def numeric_stage_values(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        stage_durations = record.get("stage_durations")
        if not isinstance(stage_durations, dict):
            continue
        for key, value in stage_durations.items():
            if isinstance(value, int | float):
                values[str(key)].append(float(value))
    return values


def summarize_stage_durations(runs: list[tuple[Path, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, records in runs:
        for key, values in sorted(numeric_stage_values(records).items()):
            rows.append(
                {
                    "run": run_name(path),
                    "stage": key,
                    "count": len(values),
                    "mean": mean(values),
                    "p50": percentile(values, 50),
                    "p90": percentile(values, 90),
                }
            )
    return rows


def parse_gpu_set(value: str) -> set[int]:
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def load_gpu_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for record in load_jsonl(path):
            record["_source"] = str(path)
            records.append(record)
    return records


def gpu_util_series(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_gpu: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        if not record.get("ok") or record.get("util_gpu_pct") is None:
            continue
        by_gpu[int(record["gpu_index"])].append((float(record["sample_ts"]), float(record["util_gpu_pct"])))
    series: list[dict[str, Any]] = []
    for gpu_index, points in sorted(by_gpu.items()):
        points.sort()
        if not points:
            continue
        t0 = points[0][0]
        series.append(
            {
                "name": f"GPU {gpu_index}",
                "x": [timestamp - t0 for timestamp, _ in points],
                "y": [util for _, util in points],
            }
        )
    return series


def phase_util_summary(
    records: list[dict[str, Any]],
    *,
    ar_gpus: set[int],
    dit_gpus: set[int],
    idle_threshold: float,
    busy_threshold: float,
) -> dict[str, Any]:
    if not ar_gpus or not dit_gpus:
        return {"available": False}
    by_ts: dict[float, dict[int, float]] = defaultdict(dict)
    for record in records:
        if not record.get("ok") or record.get("util_gpu_pct") is None:
            continue
        by_ts[float(record["sample_ts"])][int(record["gpu_index"])] = float(record["util_gpu_pct"])
    rows: list[tuple[float, float | None, float | None]] = []
    for sample_ts, values in sorted(by_ts.items()):
        ar_values = [values[gpu] for gpu in ar_gpus if gpu in values]
        dit_values = [values[gpu] for gpu in dit_gpus if gpu in values]
        ar_avg = mean(ar_values) if ar_values else None
        dit_avg = mean(dit_values) if dit_values else None
        rows.append((sample_ts, ar_avg, dit_avg))
    if len(rows) < 2:
        return {"available": False}

    bucket_s = {
        "ar_idle_dit_busy_s": 0.0,
        "dit_idle_ar_busy_s": 0.0,
        "both_busy_s": 0.0,
        "both_idle_s": 0.0,
        "observed_s": 0.0,
    }
    for idx, (sample_ts, ar_avg, dit_avg) in enumerate(rows[:-1]):
        next_ts = rows[idx + 1][0]
        duration = max(0.0, next_ts - sample_ts)
        if ar_avg is None or dit_avg is None:
            continue
        bucket_s["observed_s"] += duration
        if ar_avg < idle_threshold and dit_avg >= busy_threshold:
            bucket_s["ar_idle_dit_busy_s"] += duration
        elif dit_avg < idle_threshold and ar_avg >= busy_threshold:
            bucket_s["dit_idle_ar_busy_s"] += duration
        elif ar_avg >= busy_threshold and dit_avg >= busy_threshold:
            bucket_s["both_busy_s"] += duration
        elif ar_avg < idle_threshold and dit_avg < idle_threshold:
            bucket_s["both_idle_s"] += duration

    observed_s = bucket_s["observed_s"]
    ratios = {
        key.replace("_s", "_ratio"): (value / observed_s if observed_s > 0 else None)
        for key, value in bucket_s.items()
        if key != "observed_s"
    }
    t0 = rows[0][0]
    return {
        "available": True,
        **bucket_s,
        **ratios,
        "series": [
            {"name": "AR avg GPU util", "x": [row[0] - t0 for row in rows], "y": [row[1] for row in rows]},
            {"name": "DiT avg GPU util", "x": [row[0] - t0 for row in rows], "y": [row[2] for row in rows]},
        ],
    }


def build_summary_table(summaries: list[dict[str, Any]]) -> str:
    header = (
        "<tr><th>Run</th><th>Total</th><th>OK</th><th>Success</th><th>Throughput</th>"
        "<th>p50</th><th>p90</th><th>p99</th><th>Prompt mix</th></tr>"
    )
    rows = []
    for summary in summaries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(summary['name'])}</td>"
            f"<td>{summary['total']}</td>"
            f"<td>{summary['ok']}</td>"
            f"<td>{fmt(summary['success_rate'])}</td>"
            f"<td>{fmt(summary['throughput_req_s'])} req/s</td>"
            f"<td>{fmt(summary['latency_p50_s'])} s</td>"
            f"<td>{fmt(summary['latency_p90_s'])} s</td>"
            f"<td>{fmt(summary['latency_p99_s'])} s</td>"
            f"<td><code>{html.escape(json.dumps(summary['profile_counts'], sort_keys=True))}</code></td>"
            "</tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def build_stage_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No response stage duration fields were found.</p>"
    header = "<tr><th>Run</th><th>Stage</th><th>Count</th><th>Mean</th><th>p50</th><th>p90</th></tr>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['run'])}</td>"
            f"<td>{html.escape(row['stage'])}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{fmt(row['mean'])}</td>"
            f"<td>{fmt(row['p50'])}</td>"
            f"<td>{fmt(row['p90'])}</td>"
            "</tr>"
        )
    return f"<table>{header}{''.join(body)}</table>"


def build_html(title: str, summary_table: str, stage_table: str, report_data: dict[str, Any]) -> str:
    data_text = json.dumps(report_data, ensure_ascii=False).replace("</", "<\\/")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    code {{ font-size: 12px; }}
    canvas {{ width: 100%; height: 320px; border: 1px solid #d0d7de; }}
    .toggles {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0; font-size: 13px; }}
    .note {{ color: #5f6368; max-width: 1000px; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <p class="note">Version A measures the current separated deployment. HunyuanImage3 DiT is still request-mode
  single request, so GPU-util imbalance here is a baseline observation, not a strict stepwise continuous-batching
  comparison.</p>

  <section>
    <h2>Run Summary</h2>
    {summary_table}
  </section>

  <section>
    <h2>Latency Over Time</h2>
    <div id="latency_toggles" class="toggles"></div>
    <canvas id="latency_chart" width="1200" height="360"></canvas>
  </section>

  <section>
    <h2>Approximate Client In-flight Requests</h2>
    <div id="inflight_toggles" class="toggles"></div>
    <canvas id="inflight_chart" width="1200" height="360"></canvas>
  </section>

  <section>
    <h2>GPU Utilization</h2>
    <div id="gpu_toggles" class="toggles"></div>
    <canvas id="gpu_chart" width="1200" height="360"></canvas>
  </section>

  <section>
    <h2>AR vs DiT GPU-util Heuristic</h2>
    <p class="note">This uses nvidia-smi utilization only. It does not prove queue ownership unless server-side
    queue metrics are collected too.</p>
    <pre id="phase_summary"></pre>
    <div id="phase_toggles" class="toggles"></div>
    <canvas id="phase_chart" width="1200" height="360"></canvas>
  </section>

  <section>
    <h2>Response Stage Durations</h2>
    {stage_table}
  </section>

  <script id="report-data" type="application/json">{data_text}</script>
  <script>
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const COLORS = ["#1a73e8", "#d93025", "#188038", "#f29900", "#9334e6", "#00acc1", "#5f6368", "#c5221f"];

function finitePoints(series) {{
  const points = [];
  for (let i = 0; i < series.x.length; i++) {{
    const x = series.x[i];
    const y = series.y[i];
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  }}
  return points;
}}

function drawChart(canvasId, series, enabled) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const active = series.filter((_, idx) => enabled[idx]);
  const points = active.flatMap(finitePoints);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#d0d7de";
  ctx.strokeRect(50, 20, canvas.width - 70, canvas.height - 60);
  if (points.length === 0) {{
    ctx.fillStyle = "#5f6368";
    ctx.fillText("No data", 60, 45);
    return;
  }}
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(0, Math.min(...ys));
  const ymax = Math.max(...ys);
  const sx = (x) => 50 + ((x - xmin) / Math.max(xmax - xmin, 1e-9)) * (canvas.width - 70);
  const sy = (y) => canvas.height - 40 - ((y - ymin) / Math.max(ymax - ymin, 1e-9)) * (canvas.height - 60);
  ctx.fillStyle = "#5f6368";
  ctx.fillText(`${{xmin.toFixed(1)}}s`, 50, canvas.height - 20);
  ctx.fillText(`${{xmax.toFixed(1)}}s`, canvas.width - 80, canvas.height - 20);
  ctx.fillText(`${{ymax.toFixed(2)}}`, 8, 28);
  ctx.fillText(`${{ymin.toFixed(2)}}`, 8, canvas.height - 42);
  active.forEach((item, activeIdx) => {{
    const originalIdx = series.indexOf(item);
    const pointsForSeries = finitePoints(item);
    ctx.beginPath();
    ctx.strokeStyle = COLORS[originalIdx % COLORS.length];
    ctx.lineWidth = 2;
    pointsForSeries.forEach(([x, y], pointIdx) => {{
      if (pointIdx === 0) ctx.moveTo(sx(x), sy(y));
      else ctx.lineTo(sx(x), sy(y));
    }});
    ctx.stroke();
    ctx.fillStyle = COLORS[originalIdx % COLORS.length];
    ctx.fillText(item.name, 60 + activeIdx * 180, 16);
  }});
}}

function mountChart(chartId, togglesId, series) {{
  const enabled = series.map(() => true);
  const toggles = document.getElementById(togglesId);
  toggles.innerHTML = "";
  series.forEach((item, idx) => {{
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.onchange = () => {{
      enabled[idx] = input.checked;
      drawChart(chartId, series, enabled);
    }};
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + item.name));
    toggles.appendChild(label);
  }});
  drawChart(chartId, series, enabled);
}}

mountChart("latency_chart", "latency_toggles", DATA.charts.latency);
mountChart("inflight_chart", "inflight_toggles", DATA.charts.inflight);
mountChart("gpu_chart", "gpu_toggles", DATA.charts.gpu_util);
if (DATA.phase_util.available) {{
  document.getElementById("phase_summary").textContent = JSON.stringify(DATA.phase_util, null, 2);
  mountChart("phase_chart", "phase_toggles", DATA.phase_util.series);
}} else {{
  document.getElementById("phase_summary").textContent = "No AR/DiT GPU grouping data available.";
  mountChart("phase_chart", "phase_toggles", []);
}}
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-jsonl", action="append", required=True, help="Request result JSONL. Repeatable.")
    parser.add_argument("--gpu-jsonl", action="append", default=[], help="nvidia-smi sample JSONL. Repeatable.")
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--title", default="HunyuanImage3 Phase Imbalance Report")
    parser.add_argument("--slo-s", action="append", type=float, default=None)
    parser.add_argument("--ar-gpus", default="0,1")
    parser.add_argument("--dit-gpus", default="2,3")
    parser.add_argument("--idle-util-threshold", type=float, default=20.0)
    parser.add_argument("--busy-util-threshold", type=float, default=50.0)
    args = parser.parse_args()
    slo_values = args.slo_s if args.slo_s is not None else [120.0, 180.0, 300.0]

    run_paths = [Path(path) for path in args.result_jsonl]
    runs = [(path, load_jsonl(path)) for path in run_paths]
    summaries = [summarize_run(path, records, slo_values) for path, records in runs]
    stage_rows = summarize_stage_durations(runs)
    gpu_records = load_gpu_records([Path(path) for path in args.gpu_jsonl])
    phase_util = phase_util_summary(
        gpu_records,
        ar_gpus=parse_gpu_set(args.ar_gpus),
        dit_gpus=parse_gpu_set(args.dit_gpus),
        idle_threshold=args.idle_util_threshold,
        busy_threshold=args.busy_util_threshold,
    )
    report_data = {
        "summaries": summaries,
        "stage_duration_summary": stage_rows,
        "charts": {
            "latency": [latency_series(run_name(path), records) for path, records in runs],
            "inflight": [inflight_series(run_name(path), records) for path, records in runs],
            "gpu_util": gpu_util_series(gpu_records),
        },
        "phase_util": phase_util,
    }

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(
            args.title,
            build_summary_table(summaries),
            build_stage_table(stage_rows),
            report_data,
        ),
        encoding="utf-8",
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
