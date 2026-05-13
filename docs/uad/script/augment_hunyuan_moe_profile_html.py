#!/usr/bin/env python3
"""Build the interactive Hunyuan MoE HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-report-data", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--single-expert-csv", required=True)
    parser.add_argument("--active-expert-csv", default=None)
    parser.add_argument("--topk1-single-expert-csv", default=None)
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--skip-single-expert-profile", action="store_true")
    return parser.parse_args()


def read_csv(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def maybe_number(value: Any) -> Any:
    if value in ("", None):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: maybe_number(value) for key, value in row.items()} for row in rows]


def build_series_from_rows(rows: list[dict[str, Any]], token_key: str = "tokens") -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "label" not in row or "time_ms" not in row or token_key not in row:
            continue
        label = str(row["label"])
        series = by_label.setdefault(
            label,
            {
                "label": label,
                "category": row.get("category", "other"),
                "order": int(row.get("order", 999)),
                "kernel": row.get("kernel", ""),
                "points": [],
            },
        )
        series["points"].append({"tokens": int(row[token_key]), "ms": float(row["time_ms"])})
    for series in by_label.values():
        series["points"].sort(key=lambda p: p["tokens"])
    return sorted(by_label.values(), key=lambda s: (s["order"], s["label"]))


def event_meta_from_rows(rows: list[dict[str, Any]], token_key: str = "tokens") -> dict[str, dict[str, float]]:
    meta: dict[str, dict[str, float]] = {}
    for row in rows:
        if token_key not in row:
            continue
        token_meta = meta.setdefault(str(int(row[token_key])), {})
        for key, value in row.items():
            if key.startswith("cuda_event_") or key.endswith("_tflops") or key in {
                "hidden_size",
                "intermediate_size",
                "expert_id",
                "layer_id",
                "top_k",
                "rows_for_expert",
            }:
                if value not in ("", None):
                    token_meta[key] = maybe_number(value)
    return meta


def add_moe_throughput_series(report: dict[str, Any]) -> None:
    wanted = {
        "gemm1_gate_up_activation": "MoE GEMM1 gate/up input tokens/s",
        "gemm2_down": "MoE GEMM2 down input tokens/s",
    }
    out = []
    for category, label in wanted.items():
        by_token: dict[int, float] = {}
        for series in report.get("kernel_series", []):
            if series.get("category") != category:
                continue
            for point in series.get("points", []):
                token = int(point["tokens"])
                by_token[token] = by_token.get(token, 0.0) + float(point["ms"])
        out.append(
            {
                "label": label,
                "category": category,
                "points": [
                    {"tokens": token, "tokens_per_s": token / ms * 1000.0}
                    for token, ms in sorted(by_token.items())
                    if ms > 0
                ],
            }
        )
    if "token_meta" in report:
        out.append(
            {
                "label": "MoE FusedMoE end-to-end input tokens/s",
                "category": "moe_total",
                "points": [
                    {
                        "tokens": int(token),
                        "tokens_per_s": int(token)
                        / float(meta["cuda_event_fused_moe_ms"])
                        * 1000.0,
                    }
                    for token, meta in sorted(report["token_meta"].items(), key=lambda kv: int(kv[0]))
                    if "cuda_event_fused_moe_ms" in meta
                ],
            }
        )
    report["moe_throughput_series"] = out


def add_dense_data(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rows = numeric_rows(rows)
    report["single_expert_series"] = build_series_from_rows(rows)
    report["single_expert_meta"] = event_meta_from_rows(rows)
    wanted = {
        "dense_gemm1_gate_up": ("Dense FFN GEMM1 gate/up input tokens/s", "cuda_event_dense_gemm1_gate_up_ms"),
        "dense_gemm2_down": ("Dense FFN GEMM2 down input tokens/s", "cuda_event_dense_gemm2_down_ms"),
        "dense_silu": ("Dense FFN SiLU input tokens/s", "cuda_event_dense_silu_ms"),
        "dense_mul": ("Dense FFN multiply input tokens/s", "cuda_event_dense_mul_ms"),
    }
    series_by_category = {s["category"]: s for s in report["single_expert_series"]}
    out = []
    for category, (label, event_key) in wanted.items():
        points = []
        for token, meta in sorted(report["single_expert_meta"].items(), key=lambda kv: int(kv[0])):
            if event_key in meta and float(meta[event_key]) > 0:
                points.append({"tokens": int(token), "tokens_per_s": int(token) / float(meta[event_key]) * 1000.0})
        if not points and category in series_by_category:
            points = [
                {"tokens": p["tokens"], "tokens_per_s": p["tokens"] / p["ms"] * 1000.0}
                for p in series_by_category[category]["points"]
                if p["ms"] > 0
            ]
        out.append({"label": label, "category": category, "points": points})
    out.append(
        {
            "label": "Dense FFN end-to-end input tokens/s",
            "category": "dense_total",
            "points": [
                {"tokens": int(token), "tokens_per_s": int(token) / float(meta["cuda_event_dense_ffn_ms"]) * 1000.0}
                for token, meta in sorted(report["single_expert_meta"].items(), key=lambda kv: int(kv[0]))
                if "cuda_event_dense_ffn_ms" in meta and float(meta["cuda_event_dense_ffn_ms"]) > 0
            ],
        }
    )
    report["single_expert_throughput_series"] = out


def add_topk1_data(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rows = numeric_rows(rows)
    report["topk1_single_expert_series"] = build_series_from_rows(rows)
    report["topk1_single_expert_meta"] = event_meta_from_rows(rows)
    by_category: dict[str, dict[int, float]] = {}
    for series in report["topk1_single_expert_series"]:
        for point in series["points"]:
            by_category.setdefault(series["category"], {})
            by_category[series["category"]][int(point["tokens"])] = (
                by_category[series["category"]].get(int(point["tokens"]), 0.0) + float(point["ms"])
            )
    out = []
    for category, label in {
        "gemm1_gate_up_activation": "TopK=1 MoE GEMM1 gate/up input tokens/s",
        "gemm2_down": "TopK=1 MoE GEMM2 down input tokens/s",
    }.items():
        out.append(
            {
                "label": label,
                "category": category,
                "points": [
                    {"tokens": token, "tokens_per_s": token / ms * 1000.0}
                    for token, ms in sorted(by_category.get(category, {}).items())
                    if ms > 0
                ],
            }
        )
    out.append(
        {
            "label": "TopK=1 MoE end-to-end input tokens/s",
            "category": "topk1_total",
            "points": [
                {"tokens": int(token), "tokens_per_s": int(token) / float(meta["cuda_event_topk1_moe_ms"]) * 1000.0}
                for token, meta in sorted(report["topk1_single_expert_meta"].items(), key=lambda kv: int(kv[0]))
                if "cuda_event_topk1_moe_ms" in meta and float(meta["cuda_event_topk1_moe_ms"]) > 0
            ],
        }
    )
    report["topk1_single_expert_throughput_series"] = out


def add_active_expert_sweep(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rows = numeric_rows(rows)
    metrics = [
        ("gemm_total_tflops", "GEMM total TFLOPs"),
        ("gemm1_tflops", "GEMM1 gate/up TFLOPs"),
        ("gemm2_tflops", "GEMM2 down TFLOPs"),
        ("end_to_end_tflops", "End-to-end TFLOPs"),
        ("cuda_event_fused_moe_ms", "CUDA event latency ms"),
    ]
    series: dict[str, list[dict[str, Any]]] = {}
    keys = sorted({(int(r["top_k"]), int(r["total_token_expert_rows"])) for r in rows})
    for metric, metric_label in metrics:
        metric_series = []
        for top_k, total_rows in keys:
            points = [
                {
                    "active_experts": int(r["active_experts"]),
                    "value": float(r[metric]),
                    "rows_per_active_expert": float(r["rows_per_active_expert"]),
                    "input_tokens": int(r["input_tokens"]),
                    "top_k": top_k,
                    "total_token_expert_rows": total_rows,
                }
                for r in rows
                if int(r["top_k"]) == top_k and int(r["total_token_expert_rows"]) == total_rows
            ]
            metric_series.append(
                {
                    "label": f"top_k={top_k}, {total_rows} token-expert rows",
                    "metric": metric,
                    "metric_label": metric_label,
                    "top_k": top_k,
                    "total_token_expert_rows": total_rows,
                    "points": sorted(points, key=lambda p: p["active_experts"]),
                }
            )
        series[metric] = metric_series
    report["active_expert_sweep"] = {
        "rows": sorted(rows, key=lambda r: (int(r["top_k"]), int(r["total_token_expert_rows"]), int(r["active_experts"]))),
        "metrics": [{"key": key, "label": label} for key, label in metrics],
        "series": series,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<style>
body { font-family: Arial, sans-serif; margin: 18px; color: #111827; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.tabs button { padding: 7px 12px; border: 1px solid #9ca3af; background: #fff; border-radius: 6px; cursor: pointer; }
.tabs button.active { background: #111827; color: #fff; }
.tab { display: none; }
.tab.active { display: block; }
.layout { display: grid; grid-template-columns: minmax(720px, 1fr) 410px; gap: 16px; align-items: start; margin-bottom: 16px; }
.panel { border: 1px solid #d1d5db; border-radius: 6px; padding: 12px; background: #fff; }
.plot { width: 100%; height: 430px; border: 1px solid #e5e7eb; background: #f9fafb; }
#barPlot { height: 330px; }
.legend { max-height: 330px; overflow: auto; font-size: 12px; line-height: 1.35; }
.legend label { display: block; margin: 4px 0; white-space: nowrap; }
.inlineLegend { margin-top: 8px; max-height: none; }
.inlineLegend label { display: inline-block; margin-right: 14px; }
.muted { color: #6b7280; font-size: 12px; }
svg text { font-family: Arial, sans-serif; }
</style>
</head>
<body>
<h2>__TITLE__</h2>
<p class="muted">Layer __LAYER_ID__, expert __EXPERT_ID__. Tab2 dense FFN and Tab4 top_k=1 MoE both use the same Hunyuan expert weights and shape.</p>
<div class="tabs">
  <button class="tabBtn active" data-tab="moeTab">MoE FusedMoE</button>
  <button class="tabBtn" data-tab="denseTab">Single Expert Dense FFN</button>
  <button class="tabBtn" data-tab="activeExpertTab">Active Expert Sweep</button>
  <button class="tabBtn" data-tab="topk1Tab">TopK=1 Same Expert MoE</button>
</div>

<section id="moeTab" class="tab active">
  <div class="layout">
    <div class="panel"><h3>MoE Kernel Time</h3><svg id="moeTimePlot" class="plot"></svg></div>
    <div class="panel"><button id="moeAllBtn">All</button> <button id="moeNoneBtn">None</button> <button id="moeMainBtn">Main</button><div id="moeLegend" class="legend"></div></div>
  </div>
  <div class="layout">
    <div class="panel"><h3>MoE Input Token/s</h3><svg id="moeThroughputPlot" class="plot"></svg><div id="moeThroughputLegend" class="legend inlineLegend"></div></div>
    <div class="panel"><h3 id="barTitle">Expert token distribution</h3><svg id="barPlot" class="plot"></svg><pre id="details" class="muted"></pre></div>
  </div>
</section>

<section id="denseTab" class="tab">
  <div class="layout">
    <div class="panel"><h3>Single Expert Dense FFN Kernel Time</h3><svg id="denseTimePlot" class="plot"></svg></div>
    <div class="panel"><button id="denseAllBtn">All</button> <button id="denseNoneBtn">None</button> <button id="denseMainBtn">Main</button><div id="denseLegend" class="legend"></div></div>
  </div>
  <div class="layout">
    <div class="panel"><h3>Single Expert Dense FFN Input Token/s</h3><svg id="denseThroughputPlot" class="plot"></svg><div id="denseThroughputLegend" class="legend inlineLegend"></div></div>
    <div class="panel"><pre id="denseDetails" class="muted"></pre></div>
  </div>
</section>

<section id="activeExpertTab" class="tab">
  <div class="layout">
    <div class="panel"><h3>Fixed Token-Expert Rows vs Active Experts</h3><label class="muted">Metric <select id="activeMetric"></select></label><svg id="activeExpertPlot" class="plot"></svg></div>
    <div class="panel"><pre id="activeExpertDetails" class="muted"></pre></div>
  </div>
</section>

<section id="topk1Tab" class="tab">
  <div class="layout">
    <div class="panel"><h3>TopK=1 Same Expert MoE Kernel Time</h3><svg id="topk1TimePlot" class="plot"></svg></div>
    <div class="panel"><button id="topk1AllBtn">All</button> <button id="topk1NoneBtn">None</button> <button id="topk1MainBtn">Main</button><div id="topk1Legend" class="legend"></div></div>
  </div>
  <div class="layout">
    <div class="panel"><h3>TopK=1 Same Expert MoE Input Token/s</h3><svg id="topk1ThroughputPlot" class="plot"></svg><div id="topk1ThroughputLegend" class="legend inlineLegend"></div></div>
    <div class="panel"><pre id="topk1Details" class="muted"></pre></div>
  </div>
</section>

<script>
const report = __PAYLOAD__;
const tokens = report.tokens || [];
let selectedToken = tokens[Math.floor(tokens.length / 2)] || 1;
const activeSweep = report.active_expert_sweep || {rows: [], metrics: [], series: {}};
let activeMetric = activeSweep.metrics.length ? activeSweep.metrics[0].key : "gemm_total_tflops";
let selectedActiveExperts = 64;
function log2(x) { return Math.log(x) / Math.log(2); }
function make(tag, attrs={}, text=null) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== null) el.textContent = text;
  return el;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function color(i) {
  const colors = ["#2563eb","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2","#4f46e5","#be123c","#65a30d","#0f766e","#7c3aed","#ca8a04"];
  return colors[i % colors.length];
}
function sxFactory(w, ml, mr) {
  const min = log2(tokens[0] || 1), max = log2(tokens[tokens.length - 1] || 2);
  return x => ml + (log2(x) - min) / (max - min || 1) * (w - ml - mr);
}
function nearestTokenFromX(px, sx) {
  let best = tokens[0], bestD = Infinity;
  for (const t of tokens) {
    const d = Math.abs(sx(t) - px);
    if (d < bestD) { bestD = d; best = t; }
  }
  return best;
}
function renderSeriesPlot(svgId, series, selectedSet, yKey, yLabel, opts={}) {
  const svg = document.getElementById(svgId); clear(svg);
  const rect = svg.getBoundingClientRect();
  const w = Math.max(720, rect.width), h = 430, ml = 86, mr = 24, mt = 24, mb = 54;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  if (!series || !series.length) {
    svg.appendChild(make("text", {x:24,y:48,fontSize:14,fill:"#6b7280"}, "no data"));
    return;
  }
  const sx = sxFactory(w, ml, mr);
  let ymax = 0;
  series.forEach(s => {
    if (selectedSet && !selectedSet.has(s.label)) return;
    s.points.forEach(p => ymax = Math.max(ymax, p[yKey]));
  });
  ymax = (ymax || 1) * 1.12;
  const sy = y => mt + (1 - y / ymax) * (h - mt - mb);
  svg.appendChild(make("rect", {x:ml,y:mt,width:w-ml-mr,height:h-mt-mb,fill:"#f9fafb",stroke:"#d1d5db"}));
  for (let i=0;i<=5;i++) {
    const yv = ymax * i / 5, y = sy(yv);
    svg.appendChild(make("line", {x1:ml,y1:y,x2:w-mr,y2:y,stroke:"#e5e7eb"}));
    svg.appendChild(make("text", {x:ml-8,y:y+4,"text-anchor":"end",fontSize:12,fill:"#374151"}, opts.formatY ? opts.formatY(yv) : yv.toFixed(2)));
  }
  for (const t of tokens) {
    const x = sx(t);
    svg.appendChild(make("line", {x1:x,y1:mt,x2:x,y2:h-mb,stroke:"#eef2f7"}));
    svg.appendChild(make("text", {x:x,y:h-28,"text-anchor":"middle",fontSize:11,fill:"#374151"}, t));
  }
  series.forEach((s, i) => {
    if (selectedSet && !selectedSet.has(s.label)) return;
    const d = s.points.map((p,j) => `${j===0?"M":"L"} ${sx(p.tokens)} ${sy(p[yKey])}`).join(" ");
    svg.appendChild(make("path", {d, fill:"none", stroke:color(i), "stroke-width":2}));
  });
  const xSel = sx(selectedToken);
  svg.appendChild(make("line", {x1:xSel,y1:mt,x2:xSel,y2:h-mb,stroke:"#111827","stroke-width":1.3,"stroke-dasharray":"4 3"}));
  svg.appendChild(make("text", {x:w/2,y:h-7,"text-anchor":"middle",fontSize:13,fontWeight:700}, "input tokens (log2 scale)"));
  svg.appendChild(make("text", {x:18,y:h/2,transform:`rotate(-90 18 ${h/2})`,"text-anchor":"middle",fontSize:13,fontWeight:700}, yLabel));
  svg.onmousemove = ev => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    const next = nearestTokenFromX(loc.x, sx);
    if (next !== selectedToken) { selectedToken = next; renderAllPlots(); }
  };
}
function renderLegend(divId, series, selectedSet, rerender) {
  const div = document.getElementById(divId); div.innerHTML = "";
  (series || []).forEach((s,i) => {
    const label = document.createElement("label");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = selectedSet.has(s.label);
    cb.onchange = () => { cb.checked ? selectedSet.add(s.label) : selectedSet.delete(s.label); rerender(); };
    const sw = document.createElement("span"); sw.textContent = "■ "; sw.style.color = color(i);
    label.appendChild(cb); label.appendChild(sw); label.appendChild(document.createTextNode(s.label)); div.appendChild(label);
  });
}
const moeMainCats = new Set(["prefix_sum","expert_map_build","dispatch_expand","gemm1_gate_up_activation","gemm2_down","finalize_combine"]);
const denseMainCats = new Set(["dense_gemm1_gate_up","dense_silu","dense_mul","dense_gemm2_down"]);
const denseThroughputMainCats = new Set(["dense_gemm1_gate_up","dense_gemm2_down","dense_total"]);
const topk1MainCats = new Set(["prefix_sum","expert_map_build","dispatch_expand","gemm1_gate_up_activation","gemm2_down","finalize_combine"]);
const topk1ThroughputMainCats = new Set(["gemm1_gate_up_activation","gemm2_down","topk1_total"]);
const moeSelected = new Set((report.kernel_series || []).filter(s => moeMainCats.has(s.category)).map(s => s.label));
const denseSelected = new Set((report.single_expert_series || []).filter(s => denseMainCats.has(s.category)).map(s => s.label));
const topk1Selected = new Set((report.topk1_single_expert_series || []).filter(s => topk1MainCats.has(s.category)).map(s => s.label));
const moeThroughputSelected = new Set((report.moe_throughput_series || []).map(s => s.label));
const denseThroughputSelected = new Set((report.single_expert_throughput_series || []).filter(s => denseThroughputMainCats.has(s.category)).map(s => s.label));
const topk1ThroughputSelected = new Set((report.topk1_single_expert_throughput_series || []).filter(s => topk1ThroughputMainCats.has(s.category)).map(s => s.label));
function formatThroughput(v) {
  if (v >= 1e6) return (v/1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v/1e3).toFixed(0) + "k";
  return v.toFixed(0);
}
function renderBars() {
  const svg = document.getElementById("barPlot"); clear(svg);
  const counts = report.expert_counts ? report.expert_counts[selectedToken] : null;
  if (!counts) return;
  const rect = svg.getBoundingClientRect();
  const w = Math.max(320, rect.width), h = 330, ml = 48, mr = 16, mt = 20, mb = 38;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const ymax = Math.max(...counts) * 1.12 || 1;
  const barW = (w - ml - mr) / counts.length;
  const sy = y => mt + (1 - y / ymax) * (h - mt - mb);
  svg.appendChild(make("rect", {x:ml,y:mt,width:w-ml-mr,height:h-mt-mb,fill:"#f9fafb",stroke:"#d1d5db"}));
  counts.forEach((c,i) => svg.appendChild(make("rect", {x:ml+i*barW+1,y:sy(c),width:Math.max(1,barW-2),height:h-mb-sy(c),fill:"#2563eb"})));
  const meta = report.token_meta[selectedToken] || {};
  document.getElementById("barTitle").textContent = `Expert token distribution @ ${selectedToken} input tokens`;
  document.getElementById("details").textContent =
    `balanced rows/expert: ${(meta.mean || 0).toFixed(2)}\n` +
    `min / max: ${meta.min} / ${meta.max}\n` +
    `MoE event total: ${(meta.cuda_event_fused_moe_ms || 0).toFixed(4)} ms`;
}
function renderDenseDetails() {
  const meta = (report.single_expert_meta || {})[selectedToken] || {};
  document.getElementById("denseDetails").textContent =
    `same expert: layer ${meta.layer_id ?? "__LAYER_ID__"}, expert ${meta.expert_id ?? "__EXPERT_ID__"}\n` +
    `shape: hidden=${meta.hidden_size ?? 4096}, intermediate=${meta.intermediate_size ?? 3072}\n` +
    `dense FFN event total: ${(meta.cuda_event_dense_ffn_ms || 0).toFixed(4)} ms\n` +
    `dense FFN input tokens/s: ${meta.cuda_event_dense_ffn_ms ? (selectedToken / meta.cuda_event_dense_ffn_ms * 1000).toFixed(2) : "n/a"}`;
}
function renderTopk1Details() {
  const meta = (report.topk1_single_expert_meta || {})[selectedToken] || {};
  const dense = (report.single_expert_meta || {})[selectedToken] || {};
  document.getElementById("topk1Details").textContent =
    `same expert: layer ${meta.layer_id ?? "__LAYER_ID__"}, expert ${meta.expert_id ?? "__EXPERT_ID__"}\n` +
    `shape: hidden=${meta.hidden_size ?? 4096}, intermediate=${meta.intermediate_size ?? 3072}\n` +
    `top_k=${meta.top_k ?? 1}, rows_for_expert=${meta.rows_for_expert ?? selectedToken}\n` +
    `TopK=1 MoE event total: ${(meta.cuda_event_topk1_moe_ms || 0).toFixed(4)} ms\n` +
    `Dense FFN event total: ${(dense.cuda_event_dense_ffn_ms || 0).toFixed(4)} ms\n` +
    `TopK=1 MoE GEMM total TFLOPs: ${(meta.gemm_total_tflops || 0).toFixed(1)}\n` +
    `TopK=1 MoE end-to-end TFLOPs: ${(meta.end_to_end_tflops || 0).toFixed(1)}`;
}
function activeExpertsValues() { return [...new Set(activeSweep.rows.map(r => r.active_experts))].sort((a,b) => a-b); }
function activeMetricLabel() {
  const metric = activeSweep.metrics.find(m => m.key === activeMetric);
  return metric ? metric.label : activeMetric;
}
function renderActiveExpertPlot() {
  const svg = document.getElementById("activeExpertPlot"); clear(svg);
  const details = document.getElementById("activeExpertDetails");
  const values = activeExpertsValues();
  if (!values.length) { svg.setAttribute("viewBox", "0 0 720 430"); svg.appendChild(make("text", {x:24,y:48,fontSize:14,fill:"#6b7280"}, "no active expert sweep data")); return; }
  if (!values.includes(selectedActiveExperts)) selectedActiveExperts = values[values.length - 1];
  const series = activeSweep.series[activeMetric] || [];
  const rect = svg.getBoundingClientRect();
  const w = Math.max(720, rect.width), h = 430, ml = 86, mr = 24, mt = 24, mb = 54;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const min = Math.log2(values[0]), max = Math.log2(values[values.length - 1]);
  const sx = x => ml + (Math.log2(x) - min) / (max - min || 1) * (w - ml - mr);
  let ymax = 0; series.forEach(s => s.points.forEach(p => ymax = Math.max(ymax, p.value))); ymax = (ymax || 1) * 1.12;
  const sy = y => mt + (1 - y / ymax) * (h - mt - mb);
  svg.appendChild(make("rect", {x:ml,y:mt,width:w-ml-mr,height:h-mt-mb,fill:"#f9fafb",stroke:"#d1d5db"}));
  for (let i=0;i<=5;i++) {
    const yv = ymax * i / 5, y = sy(yv);
    svg.appendChild(make("line", {x1:ml,y1:y,x2:w-mr,y2:y,stroke:"#e5e7eb"}));
    svg.appendChild(make("text", {x:ml-8,y:y+4,"text-anchor":"end",fontSize:12,fill:"#374151"}, yv.toFixed(activeMetric.endsWith("_ms") ? 2 : 1)));
  }
  values.forEach(v => { const x = sx(v); svg.appendChild(make("text", {x:x,y:h-28,"text-anchor":"middle",fontSize:12,fill:"#374151"}, v)); });
  series.forEach((s, i) => {
    const d = s.points.map((p,j) => `${j===0?"M":"L"} ${sx(p.active_experts)} ${sy(p.value)}`).join(" ");
    svg.appendChild(make("path", {d, fill:"none", stroke:color(i), "stroke-width":2}));
    const last = s.points[s.points.length - 1];
    if (last) svg.appendChild(make("text", {x:sx(last.active_experts)+5,y:sy(last.value)+4,fontSize:12,fill:color(i)}, s.label));
  });
  const xSel = sx(selectedActiveExperts);
  svg.appendChild(make("line", {x1:xSel,y1:mt,x2:xSel,y2:h-mb,stroke:"#111827","stroke-width":1.3,"stroke-dasharray":"4 3"}));
  svg.appendChild(make("text", {x:w/2,y:h-7,"text-anchor":"middle",fontSize:13,fontWeight:700}, "active experts (log2 scale)"));
  svg.appendChild(make("text", {x:18,y:h/2,transform:`rotate(-90 18 ${h/2})`,"text-anchor":"middle",fontSize:13,fontWeight:700}, activeMetricLabel()));
  svg.onmousemove = ev => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    let best = values[0], bestD = Infinity;
    for (const v of values) { const d = Math.abs(sx(v)-loc.x); if (d < bestD) { bestD=d; best=v; } }
    if (best !== selectedActiveExperts) { selectedActiveExperts = best; renderActiveExpertPlot(); }
  };
  const rows = activeSweep.rows.filter(r => r.active_experts === selectedActiveExperts);
  details.textContent = [
    `selected active experts: ${selectedActiveExperts}`,
    `metric: ${activeMetricLabel()}`,
    "",
    "top_k  total_rows  input_tokens  rows/expert  gemm_total_TF  e2e_TF  event_ms",
    ...rows.map(r => `${String(r.top_k).padStart(5)}  ${String(r.total_token_expert_rows).padStart(10)}  ${String(r.input_tokens).padStart(12)}  ${String(r.rows_per_active_expert.toFixed(1)).padStart(11)}  ${String(r.gemm_total_tflops.toFixed(1)).padStart(13)}  ${String(r.end_to_end_tflops.toFixed(1)).padStart(6)}  ${String(r.cuda_event_fused_moe_ms.toFixed(3)).padStart(8)}`),
  ].join("\n");
}
function renderAllPlots() {
  renderSeriesPlot("moeTimePlot", report.kernel_series || [], moeSelected, "ms", "kernel time (ms)");
  renderSeriesPlot("moeThroughputPlot", report.moe_throughput_series || [], moeThroughputSelected, "tokens_per_s", "input tokens/s", {formatY: formatThroughput});
  renderSeriesPlot("denseTimePlot", report.single_expert_series || [], denseSelected, "ms", "kernel time (ms)");
  renderSeriesPlot("denseThroughputPlot", report.single_expert_throughput_series || [], denseThroughputSelected, "tokens_per_s", "input tokens/s", {formatY: formatThroughput});
  renderSeriesPlot("topk1TimePlot", report.topk1_single_expert_series || [], topk1Selected, "ms", "kernel time (ms)");
  renderSeriesPlot("topk1ThroughputPlot", report.topk1_single_expert_throughput_series || [], topk1ThroughputSelected, "tokens_per_s", "input tokens/s", {formatY: formatThroughput});
  renderBars(); renderDenseDetails(); renderTopk1Details(); renderActiveExpertPlot();
}
const activeMetricSelect = document.getElementById("activeMetric");
activeSweep.metrics.forEach(metric => { const option = document.createElement("option"); option.value = metric.key; option.textContent = metric.label; activeMetricSelect.appendChild(option); });
activeMetricSelect.value = activeMetric;
activeMetricSelect.onchange = () => { activeMetric = activeMetricSelect.value; renderActiveExpertPlot(); };
renderLegend("moeLegend", report.kernel_series || [], moeSelected, renderAllPlots);
renderLegend("denseLegend", report.single_expert_series || [], denseSelected, renderAllPlots);
renderLegend("topk1Legend", report.topk1_single_expert_series || [], topk1Selected, renderAllPlots);
renderLegend("moeThroughputLegend", report.moe_throughput_series || [], moeThroughputSelected, renderAllPlots);
renderLegend("denseThroughputLegend", report.single_expert_throughput_series || [], denseThroughputSelected, renderAllPlots);
renderLegend("topk1ThroughputLegend", report.topk1_single_expert_throughput_series || [], topk1ThroughputSelected, renderAllPlots);
document.getElementById("moeAllBtn").onclick = () => { (report.kernel_series || []).forEach(s => moeSelected.add(s.label)); renderLegend("moeLegend", report.kernel_series || [], moeSelected, renderAllPlots); renderAllPlots(); };
document.getElementById("moeNoneBtn").onclick = () => { moeSelected.clear(); renderLegend("moeLegend", report.kernel_series || [], moeSelected, renderAllPlots); renderAllPlots(); };
document.getElementById("moeMainBtn").onclick = () => { moeSelected.clear(); (report.kernel_series || []).forEach(s => { if (moeMainCats.has(s.category)) moeSelected.add(s.label); }); renderLegend("moeLegend", report.kernel_series || [], moeSelected, renderAllPlots); renderAllPlots(); };
document.getElementById("denseAllBtn").onclick = () => { (report.single_expert_series || []).forEach(s => denseSelected.add(s.label)); renderLegend("denseLegend", report.single_expert_series || [], denseSelected, renderAllPlots); renderAllPlots(); };
document.getElementById("denseNoneBtn").onclick = () => { denseSelected.clear(); renderLegend("denseLegend", report.single_expert_series || [], denseSelected, renderAllPlots); renderAllPlots(); };
document.getElementById("denseMainBtn").onclick = () => { denseSelected.clear(); (report.single_expert_series || []).forEach(s => { if (denseMainCats.has(s.category)) denseSelected.add(s.label); }); renderLegend("denseLegend", report.single_expert_series || [], denseSelected, renderAllPlots); renderAllPlots(); };
document.getElementById("topk1AllBtn").onclick = () => { (report.topk1_single_expert_series || []).forEach(s => topk1Selected.add(s.label)); renderLegend("topk1Legend", report.topk1_single_expert_series || [], topk1Selected, renderAllPlots); renderAllPlots(); };
document.getElementById("topk1NoneBtn").onclick = () => { topk1Selected.clear(); renderLegend("topk1Legend", report.topk1_single_expert_series || [], topk1Selected, renderAllPlots); renderAllPlots(); };
document.getElementById("topk1MainBtn").onclick = () => { topk1Selected.clear(); (report.topk1_single_expert_series || []).forEach(s => { if (topk1MainCats.has(s.category)) topk1Selected.add(s.label); }); renderLegend("topk1Legend", report.topk1_single_expert_series || [], topk1Selected, renderAllPlots); renderAllPlots(); };
document.querySelectorAll(".tabBtn").forEach(btn => btn.onclick = () => {
  document.querySelectorAll(".tabBtn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active"); document.getElementById(btn.dataset.tab).classList.add("active"); renderAllPlots();
});
renderAllPlots();
</script>
</body>
</html>
"""


def build_html(report: dict[str, Any], layer_id: int, expert_id: int) -> str:
    title = html.escape(str(report.get("title", "HunyuanImage3 MoE Kernel Profile")))
    return (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__PAYLOAD__", json.dumps(report))
        .replace("__LAYER_ID__", str(layer_id))
        .replace("__EXPERT_ID__", str(expert_id))
    )


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.moe_report_data).read_text())
    add_moe_throughput_series(report)
    dense_rows = read_csv(args.single_expert_csv)
    if dense_rows:
        add_dense_data(report, dense_rows)
    active_rows = read_csv(args.active_expert_csv)
    if active_rows:
        add_active_expert_sweep(report, active_rows)
    topk1_rows = read_csv(args.topk1_single_expert_csv)
    if topk1_rows:
        add_topk1_data(report, topk1_rows)
    output = Path(args.output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(report, args.layer_id, args.expert_id))
    (output.parent / "report_data_augmented.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
