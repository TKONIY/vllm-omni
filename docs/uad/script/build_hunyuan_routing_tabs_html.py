#!/usr/bin/env python3
# ruff: noqa: E501
"""Combine Hunyuan routing reports into one tabbed HTML page."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Tab label and route_report.json path in the form 'Label=/path/to/route_report.json'.",
    )
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "report"


def load_reports(specs: list[str], output_dir: Path) -> list[dict[str, Any]]:
    reports = []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--report must be Label=/path/to/json, got: {spec}")
        label, path_text = spec.split("=", 1)
        path = Path(path_text).resolve()
        report = json.loads(path.read_text())
        report.setdefault("metadata", {})
        image_url = report["metadata"].get("image_url") or ""
        if image_url:
            src = (path.parent / image_url).resolve()
            if src.exists():
                dst_name = f"{slugify(label)}-{src.name}"
                dst = image_dir / dst_name
                shutil.copy2(src, dst)
                report["metadata"]["image_url"] = f"images/{dst_name}"
        reports.append({"label": label, "path": str(path), "report": report})
    return reports


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hunyuan MoE Routing Profiles</title>
<style>
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }
header { padding: 24px 28px 16px; background: #fff; border-bottom: 1px solid #e5e7eb; }
h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }
h2 { margin: 0 0 10px; font-size: 18px; }
.muted { color: #6b7280; font-size: 13px; line-height: 1.45; }
.wrap { padding: 18px 28px 28px; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 14px; }
.tab { border: 1px solid #cbd5e1; background: #fff; color: #111827; border-radius: 6px; padding: 8px 11px; font-size: 13px; cursor: pointer; }
.tab.active { background: #111827; border-color: #111827; color: #fff; }
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
  <h1>Hunyuan MoE Routing Profiles</h1>
  <div class="muted" id="summary"></div>
  <div class="muted">Each tab uses the same routing plot: expert id on x, selected modality/phase distributions on y. Percent normalizes each selected series independently.</div>
</header>
<main class="wrap">
  <div class="tabs" id="tabs"></div>
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
<script id="reports-data" type="application/json">__REPORTS_JSON__</script>
<script>
const tabsData = JSON.parse(document.getElementById("reports-data").textContent).reports;
const colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c"];
const metricSelect = document.getElementById("metric");
const layerSelect = document.getElementById("layer");
const checks = document.getElementById("checks");
const svg = document.getElementById("plot");
const legend = document.getElementById("legend");
const tabsEl = document.getElementById("tabs");
const metricLabels = {percent: "route share %", counts: "top-k route assignments", tokens: "token positions"};
let activeIndex = 0;
let report = tabsData[0].report;
const state = {metric: "percent", layer: "global", selected: new Set()};

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
  for (const stage of Object.values(report.stages || {})) {
    const bucket = layer === "global" ? stage.global : (stage.by_layer[layer] || {});
    for (const [modality, entry] of Object.entries(bucket || {})) out[modality] = entry;
  }
  return out;
}
function availableModalities() {
  const seen = new Set();
  for (const layer of ["global", ...(report.layers || [])]) {
    for (const modality of Object.keys(modalityEntries(layer))) seen.add(modality);
  }
  const order = ["ar_text", "dit_text", "dit_image", "dit_timestep", "dit_cond_image", "dit_other", "dit_unknown", "a13b_prefill", "a13b_decode"];
  return [...seen].sort((a, b) => {
    const ia = order.indexOf(a), ib = order.indexOf(b);
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib) || a.localeCompare(b);
  });
}
function defaultSelected() {
  const mods = availableModalities();
  if (mods.includes("ar_text") && mods.includes("dit_image")) return new Set(["ar_text", "dit_image"]);
  if (mods.includes("a13b_prefill") && mods.includes("a13b_decode")) return new Set(["a13b_prefill", "a13b_decode"]);
  return new Set(mods.slice(0, 3));
}
function initTabs() {
  tabsEl.innerHTML = "";
  tabsData.forEach((tab, idx) => {
    const btn = document.createElement("button");
    btn.className = "tab" + (idx === activeIndex ? " active" : "");
    btn.textContent = tab.label;
    btn.onclick = () => switchTab(idx);
    tabsEl.appendChild(btn);
  });
}
function initControls() {
  metricSelect.innerHTML = "";
  layerSelect.innerHTML = "";
  checks.innerHTML = "";
  for (const key of ["percent", "counts", "tokens"]) {
    metricSelect.appendChild(new Option(metricLabels[key], key, key === state.metric, key === state.metric));
  }
  layerSelect.appendChild(new Option("Global", "global", state.layer === "global", state.layer === "global"));
  for (const layer of report.layers || []) layerSelect.appendChild(new Option("Layer " + layer, layer, state.layer === layer, state.layer === layer));
  metricSelect.onchange = () => { state.metric = metricSelect.value; draw(); };
  layerSelect.onchange = () => { state.layer = layerSelect.value; draw(); };
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
    drawSummary(bucket);
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
  legend.innerHTML = series.map(s => `<span><span class="swatch" style="background:${s.color}"></span>${s.label}</span>`).join("");
  drawSummary(bucket);
}
function drawSummary(bucket) {
  const rows = Object.entries(bucket).map(([modality, entry]) => {
    const totalRoutes = (entry.route_counts || []).reduce((a, b) => a + b, 0);
    const maxValue = Math.max(...(entry.route_counts || [0]));
    const maxExpert = (entry.route_counts || []).indexOf(maxValue);
    return `<tr><td>${report.modality_labels[modality] || modality}</td><td>${fmt(entry.token_positions || 0, "counts")}</td><td>${fmt(totalRoutes, "counts")}</td><td>${maxExpert}</td><td>${fmt(maxValue, "counts")}</td></tr>`;
  }).join("");
  document.getElementById("summaryTable").innerHTML = `<table><thead><tr><th>modality/phase</th><th>token positions over MoE calls</th><th>top-k assignments</th><th>top expert</th><th>top expert routes</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function initMeta() {
  const m = report.metadata || {};
  const label = tabsData[activeIndex].label;
  const imageText = m.width && m.height ? ` | image=${m.width}x${m.height}` : "";
  const stepText = m.steps ? ` | steps=${m.steps}` : "";
  document.getElementById("summary").textContent = `${label} | prompt="${m.prompt || ""}"${imageText}${stepText} | rank_policy=${m.rank_policy || ""}`;
  document.getElementById("runMeta").innerHTML = [
    `<b>Model</b>: ${m.model || ""}`,
    `<b>Stage/config</b>: ${m.stage_config || ""}`,
    `<b>Generated</b>: ${report.generated_at}`,
    `<b>Trace dir</b>: ${m.trace_dir || ""}`,
    `<b>Report</b>: ${tabsData[activeIndex].path || ""}`,
  ].join("<br>");
  document.getElementById("imageBox").innerHTML = m.image_url ? `<img class="generated" src="${m.image_url}" alt="generated image">` : "";
  document.getElementById("sources").textContent = JSON.stringify(report.selected_sources || [], null, 2);
}
function switchTab(idx) {
  activeIndex = idx;
  report = tabsData[idx].report;
  state.layer = "global";
  state.selected = defaultSelected();
  initTabs();
  initMeta();
  initControls();
  draw();
}
switchTab(0);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    reports = load_reports(args.report, output_html.parent)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }
    report_json = json.dumps(payload, ensure_ascii=False, indent=2)
    output_html.write_text(
        HTML_TEMPLATE.replace("__REPORTS_JSON__", report_json.replace("</", "<\\/")),
        encoding="utf-8",
    )
    if args.output_json:
        Path(args.output_json).write_text(report_json, encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
