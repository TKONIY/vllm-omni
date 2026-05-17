#!/usr/bin/env python3
# ruff: noqa: E501
"""Build an HTML page for real HunyuanImage3 request MoE routing traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODALITY_LABELS = {
    "ar_text": "AR text tokens",
    "dit_text": "DiT text/context tokens",
    "dit_image": "DiT image tokens",
    "dit_timestep": "DiT timestep tokens",
    "dit_cond_image": "DiT conditioning image tokens",
    "dit_other": "DiT other tokens",
    "dit_unknown": "DiT unknown tokens",
    "a13b_prefill": "Hunyuan-A13B prefill tokens",
    "a13b_decode": "Hunyuan-A13B decode tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--stage-config", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-url", default="")
    parser.add_argument(
        "--rank-policy",
        choices=["rank0", "all"],
        default="rank0",
        help="Use one representative rank per stage by default to avoid TP/EP duplicate counts.",
    )
    return parser.parse_args()


def load_traces(trace_dir: Path) -> list[dict[str, Any]]:
    traces = []
    for path in sorted(trace_dir.glob("hunyuan_moe_route_trace_*.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        if payload.get("stages"):
            traces.append(payload)
    return traces


def rank_value(metadata: dict[str, Any]) -> int:
    for key in ("dist_rank", "rank", "local_rank", "cuda_current_device"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 1_000_000


def selected_stage_entries(traces: list[dict[str, Any]], rank_policy: str) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for trace in traces:
        metadata = trace.get("metadata", {})
        for stage, stage_entry in trace.get("stages", {}).items():
            grouped.setdefault(stage, []).append((metadata | {"path": trace["_path"]}, stage_entry))

    if rank_policy == "all":
        return grouped

    selected: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for stage, entries in grouped.items():
        selected[stage] = [min(entries, key=lambda item: (rank_value(item[0]), item[0].get("pid", 0)))]
    return selected


def empty_entry(num_experts: int) -> dict[str, Any]:
    return {
        "route_counts": [0 for _ in range(num_experts)],
        "token_positions": 0,
        "topk_assignments": 0,
        "calls": 0,
    }


def merge_modality_entry(dst: dict[str, Any], src: dict[str, Any], num_experts: int) -> None:
    counts = src.get("route_counts", [])
    if len(dst["route_counts"]) < num_experts:
        dst["route_counts"].extend([0] * (num_experts - len(dst["route_counts"])))
    for idx, value in enumerate(counts):
        dst["route_counts"][idx] += int(value)
    dst["token_positions"] += int(src.get("token_positions", 0) or 0)
    dst["topk_assignments"] += int(src.get("topk_assignments", 0) or 0)
    dst["calls"] += int(src.get("calls", 0) or 0)


def merge_bucket(dst: dict[str, Any], src: dict[str, Any], num_experts: int) -> None:
    for modality, entry in src.items():
        out = dst.setdefault(modality, empty_entry(num_experts))
        merge_modality_entry(out, entry, num_experts)


def build_report(args: argparse.Namespace, traces: list[dict[str, Any]]) -> dict[str, Any]:
    selected = selected_stage_entries(traces, args.rank_policy)
    stages: dict[str, Any] = {}
    selected_sources = []
    layer_ids: set[str] = set()

    for stage, entries in selected.items():
        num_experts = max(int(entry.get("num_experts", 0) or 0) for _, entry in entries)
        stage_out = {
            "num_experts": num_experts,
            "top_k": max(int(entry.get("top_k", 0) or 0) for _, entry in entries),
            "num_calls": sum(int(entry.get("num_calls", 0) or 0) for _, entry in entries),
            "global": {},
            "by_layer": {},
        }
        for metadata, entry in entries:
            selected_sources.append(
                {
                    "stage": stage,
                    "path": metadata.get("path", ""),
                    "pid": metadata.get("pid"),
                    "dist_rank": metadata.get("dist_rank"),
                    "rank": metadata.get("rank"),
                    "local_rank": metadata.get("local_rank"),
                    "tp_rank": metadata.get("tp_rank"),
                    "ep_rank": metadata.get("ep_rank"),
                    "cuda_current_device": metadata.get("cuda_current_device"),
                    "cuda_visible_devices": metadata.get("cuda_visible_devices"),
                }
            )
            merge_bucket(stage_out["global"], entry.get("global", {}), num_experts)
            for layer_id, bucket in entry.get("by_layer", {}).items():
                layer_ids.add(str(layer_id))
                merge_bucket(stage_out["by_layer"].setdefault(str(layer_id), {}), bucket, num_experts)
        stages[stage] = stage_out

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "prompt": args.prompt,
            "model": args.model,
            "stage_config": args.stage_config,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "image_url": args.image_url,
            "rank_policy": args.rank_policy,
            "trace_dir": str(Path(args.trace_dir).resolve()),
        },
        "modality_labels": MODALITY_LABELS,
        "stages": stages,
        "layers": sorted(layer_ids, key=lambda value: int(value)),
        "selected_sources": selected_sources,
        "all_trace_files": [trace["_path"] for trace in traces],
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HunyuanImage3 Real Request MoE Routing</title>
<style>
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }
header { padding: 24px 28px 16px; background: #fff; border-bottom: 1px solid #e5e7eb; }
h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }
h2 { margin: 0 0 10px; font-size: 18px; }
h3 { margin: 0 0 8px; font-size: 15px; }
.muted { color: #6b7280; font-size: 13px; line-height: 1.45; }
.wrap { padding: 18px 28px 28px; }
.grid { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(340px, 0.85fr); gap: 14px; align-items: start; }
.panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
select, label.checkbox { border: 1px solid #cbd5e1; background: #fff; color: #111827; border-radius: 6px; padding: 7px 10px; font-size: 13px; }
label.checkbox { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
.plot { width: 100%; height: 510px; display: block; background: #fbfdff; border: 1px solid #eef2f7; border-radius: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 12px; color: #374151; }
.swatch { width: 11px; height: 11px; display: inline-block; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 12px; }
img.generated { width: 100%; max-height: 340px; object-fit: contain; background: #f3f4f6; border-radius: 6px; border: 1px solid #e5e7eb; }
@media (max-width: 980px) { .grid { grid-template-columns: 1fr; } header, .wrap { padding-left: 16px; padding-right: 16px; } }
</style>
</head>
<body>
<header>
  <h1>HunyuanImage3 Real Request MoE Routing</h1>
  <div class="muted" id="summary"></div>
  <div class="muted">Default view uses normalized expert-route percentages, so AR text and DiT image distributions can be compared in one plot even though their raw token counts differ.</div>
</header>
<main class="wrap">
  <div class="grid">
    <section class="panel">
      <h2>Expert Routing Distribution</h2>
      <div class="controls">
        <label>Metric <select id="metric"></select></label>
        <label>Layer <select id="layer"></select></label>
        <span id="checks"></span>
      </div>
      <svg id="plot" class="plot"></svg>
      <div id="legend" class="legend"></div>
    </section>
    <aside class="panel">
      <h2>Run</h2>
      <div id="runMeta" class="muted"></div>
      <div id="imageBox" style="margin-top:12px"></div>
    </aside>
  </div>
  <section class="panel" style="margin-top:14px">
    <h2>Counts</h2>
    <div id="summaryTable"></div>
  </section>
  <section class="panel" style="margin-top:14px">
    <h2>Trace Sources</h2>
    <pre id="sources"></pre>
  </section>
</main>
<script id="report-data" type="application/json">__REPORT_JSON__</script>
<script>
const report = JSON.parse(document.getElementById("report-data").textContent);
const colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c"];
const metricSelect = document.getElementById("metric");
const layerSelect = document.getElementById("layer");
const checks = document.getElementById("checks");
const svg = document.getElementById("plot");
const legend = document.getElementById("legend");
const metricLabels = {percent: "route share %", counts: "top-k route assignments", tokens: "token positions"};
const state = {metric: "percent", layer: "global", selected: new Set(["ar_text", "dit_image"])};

function fmt(value, metric) {
  if (!Number.isFinite(value)) return "";
  if (metric === "percent") return value.toFixed(2) + "%";
  if (Math.abs(value) >= 1e9) return (value / 1e9).toFixed(2) + "B";
  if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(2) + "M";
  if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(1) + "K";
  return value.toFixed(0);
}
function make(tag, attrs, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}
function modalityEntries(layer) {
  const out = {};
  for (const [stageName, stage] of Object.entries(report.stages)) {
    const bucket = layer === "global" ? stage.global : (stage.by_layer[layer] || {});
    for (const [modality, entry] of Object.entries(bucket)) {
      out[modality] = entry;
    }
  }
  return out;
}
function availableModalities() {
  const seen = new Set();
  for (const layer of ["global", ...report.layers]) {
    for (const modality of Object.keys(modalityEntries(layer))) seen.add(modality);
  }
  return [...seen].sort((a, b) => {
    const order = ["ar_text", "dit_text", "dit_image", "dit_timestep", "dit_cond_image", "dit_other", "dit_unknown"];
    return order.indexOf(a) - order.indexOf(b);
  });
}
function initControls() {
  for (const key of ["percent", "counts", "tokens"]) {
    metricSelect.appendChild(new Option(metricLabels[key], key, key === state.metric, key === state.metric));
  }
  layerSelect.appendChild(new Option("Global", "global", true, true));
  for (const layer of report.layers) layerSelect.appendChild(new Option("Layer " + layer, layer));
  metricSelect.onchange = () => { state.metric = metricSelect.value; draw(); };
  layerSelect.onchange = () => { state.layer = layerSelect.value; draw(); };
  checks.innerHTML = "";
  for (const modality of availableModalities()) {
    const label = document.createElement("label");
    label.className = "checkbox";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selected.has(modality);
    input.onchange = () => {
      if (input.checked) state.selected.add(modality);
      else state.selected.delete(modality);
      draw();
    };
    label.appendChild(input);
    label.appendChild(document.createTextNode(report.modality_labels[modality] || modality));
    checks.appendChild(label);
  }
}
function valuesFor(entry, metric) {
  if (!entry) return [];
  const counts = entry.route_counts || [];
  if (metric === "counts") return counts.slice();
  const total = counts.reduce((a, b) => a + b, 0);
  if (metric === "percent") return counts.map(v => total ? v / total * 100 : 0);
  if (metric === "tokens") {
    const topk = total && entry.token_positions ? total / entry.token_positions : 1;
    return counts.map(v => topk ? v / topk : 0);
  }
  return counts.slice();
}
function draw() {
  svg.replaceChildren();
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 510;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const margin = {left: 68, right: 18, top: 20, bottom: 54};
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const bucket = modalityEntries(state.layer);
  const selected = [...state.selected].filter(m => bucket[m]);
  if (!selected.length) {
    svg.appendChild(make("text", {x: 24, y: 42, fill: "#6b7280", "font-size": 14}, "No selected modality has data for this layer."));
    legend.innerHTML = "";
    return;
  }
  const numExperts = Math.max(...selected.map(m => (bucket[m].route_counts || []).length));
  const series = selected.map((modality, i) => ({
    modality,
    label: report.modality_labels[modality] || modality,
    color: colors[i % colors.length],
    values: valuesFor(bucket[modality], state.metric),
    raw: bucket[modality],
  }));
  const ymax = Math.max(1e-9, ...series.flatMap(s => s.values));
  const xStep = plotW / numExperts;
  const barGap = 1;
  const groupW = Math.max(2, xStep * 0.82);
  const barW = Math.max(1, (groupW - barGap * (series.length - 1)) / series.length);
  const y = v => margin.top + plotH - (v / ymax) * plotH;
  svg.appendChild(make("line", {x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH, stroke: "#94a3b8"}));
  svg.appendChild(make("line", {x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH, stroke: "#94a3b8"}));
  for (let t = 0; t <= 5; t++) {
    const value = ymax * t / 5;
    const yy = y(value);
    svg.appendChild(make("line", {x1: margin.left, y1: yy, x2: margin.left + plotW, y2: yy, stroke: "#e5e7eb"}));
    svg.appendChild(make("text", {x: margin.left - 8, y: yy + 4, "text-anchor": "end", fill: "#475569", "font-size": 11}, fmt(value, state.metric)));
  }
  for (let expert = 0; expert < numExperts; expert++) {
    const gx = margin.left + expert * xStep + (xStep - groupW) / 2;
    if (expert % 4 === 0) {
      svg.appendChild(make("text", {x: margin.left + expert * xStep + xStep / 2, y: margin.top + plotH + 18, "text-anchor": "middle", fill: "#475569", "font-size": 10}, expert));
    }
    for (let i = 0; i < series.length; i++) {
      const value = series[i].values[expert] || 0;
      const yy = y(value);
      const rect = make("rect", {
        x: gx + i * (barW + barGap),
        y: yy,
        width: barW,
        height: Math.max(0, margin.top + plotH - yy),
        fill: series[i].color,
        opacity: 0.88,
      });
      rect.appendChild(make("title", {}, `${series[i].label}\nexpert=${expert}\n${metricLabels[state.metric]}=${fmt(value, state.metric)}\nraw routes=${fmt((series[i].raw.route_counts || [])[expert] || 0, "counts")}`));
      svg.appendChild(rect);
    }
  }
  svg.appendChild(make("text", {x: margin.left + plotW / 2, y: height - 16, "text-anchor": "middle", fill: "#334155", "font-size": 12}, "expert id"));
  svg.appendChild(make("text", {x: 18, y: margin.top + plotH / 2, transform: `rotate(-90 18 ${margin.top + plotH / 2})`, "text-anchor": "middle", fill: "#334155", "font-size": 12}, metricLabels[state.metric]));
  legend.innerHTML = series.map((s, i) => `<span><span class="swatch" style="background:${s.color}"></span>${s.label}</span>`).join("");
  drawSummary(bucket);
}
function drawSummary(bucket) {
  const rows = Object.entries(bucket).map(([modality, entry]) => {
    const totalRoutes = (entry.route_counts || []).reduce((a, b) => a + b, 0);
    const maxValue = Math.max(...(entry.route_counts || [0]));
    const maxExpert = (entry.route_counts || []).indexOf(maxValue);
    return `<tr><td>${report.modality_labels[modality] || modality}</td><td>${fmt(entry.token_positions || 0, "counts")}</td><td>${fmt(totalRoutes, "counts")}</td><td>${maxExpert}</td><td>${fmt(maxValue, "counts")}</td></tr>`;
  }).join("");
  document.getElementById("summaryTable").innerHTML = `<table><thead><tr><th>modality</th><th>token positions over MoE calls</th><th>top-k assignments</th><th>top expert</th><th>top expert routes</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function initMeta() {
  const m = report.metadata;
  document.getElementById("summary").textContent = `prompt="${m.prompt}" | image=${m.width}x${m.height} | steps=${m.steps} | guidance=${m.guidance_scale} | rank_policy=${m.rank_policy}`;
  document.getElementById("runMeta").innerHTML = [
    `<b>Model</b>: ${m.model || ""}`,
    `<b>Stage config</b>: ${m.stage_config || ""}`,
    `<b>Seed</b>: ${m.seed}`,
    `<b>Generated</b>: ${report.generated_at}`,
    `<b>Trace dir</b>: ${m.trace_dir}`,
  ].join("<br>");
  if (m.image_url) {
    document.getElementById("imageBox").innerHTML = `<img class="generated" src="${m.image_url}" alt="generated image">`;
  }
  document.getElementById("sources").textContent = JSON.stringify(report.selected_sources, null, 2);
}
initMeta();
initControls();
draw();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    traces = load_traces(Path(args.trace_dir))
    if not traces:
        raise SystemExit(f"No trace files found in {args.trace_dir}")
    report = build_report(args, traces)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    output_html.write_text(
        HTML_TEMPLATE.replace("__REPORT_JSON__", report_json.replace("</", "<\\/")),
        encoding="utf-8",
    )
    if args.output_json:
        Path(args.output_json).write_text(report_json, encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
