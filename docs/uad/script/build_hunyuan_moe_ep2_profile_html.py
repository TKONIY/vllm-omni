#!/usr/bin/env python3
"""Build a self-contained HTML report for the EP=2 MoE backend sweep."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPONENTS = [
    {"key": "e2e", "label": "end-to-end"},
    {"key": "gemm1", "label": "GEMM1 gate/up"},
    {"key": "gemm2", "label": "GEMM2 down"},
    {"key": "comm", "label": "cross-device reduce"},
    {"key": "routing", "label": "routing/prefix/dispatch"},
    {"key": "finalize", "label": "finalize/combine"},
    {"key": "profiler_sum", "label": "profiler kernel sum"},
]

MODE_LABELS = {
    "topk8_balanced": "EP=2 TopK=8 balanced MoE",
    "topk1_single": "EP=2 TopK=1 same expert",
}

MODE_NOTES = {
    "topk8_balanced": (
        "每个 input token 路由到 8 个 expert；64 个 expert 的 token-expert rows "
        "严格均匀。EP=2 时每 rank 持有 32 个 expert。"
    ),
    "topk1_single": (
        "所有 input token 都路由到同一个 Hunyuan expert。该 expert 默认在 rank0，"
        "所以 rank1 主要参与跨卡输出 reduce；这用于对比 MoE backend 的 top_k=1 路径。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-html", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def max_category_ms(ranks: list[dict[str, Any]], category: str) -> float:
    return max((float(rank.get("category_ms", {}).get(category, 0.0) or 0.0) for rank in ranks), default=0.0)


def component_ms(result: dict[str, Any]) -> dict[str, float]:
    ranks = result.get("ranks", [])
    routing = (
        max_category_ms(ranks, "copy_cast")
        + max_category_ms(ranks, "prefix_sum")
        + max_category_ms(ranks, "expert_map_build")
        + max_category_ms(ranks, "dispatch_expand")
        + max_category_ms(ranks, "memcpy")
        + max_category_ms(ranks, "memset")
    )
    return {
        "e2e": float(result.get("event_ms_max", 0.0) or 0.0),
        "gemm1": max_category_ms(ranks, "gemm1_gate_up_activation"),
        "gemm2": max_category_ms(ranks, "gemm2_down"),
        "comm": max_category_ms(ranks, "comm"),
        "routing": routing,
        "finalize": max_category_ms(ranks, "finalize_combine"),
        "profiler_sum": max((float(rank.get("profiler_sum_ms", 0.0) or 0.0) for rank in ranks), default=0.0),
    }


def flops_for_component(component: str, tokens: int, top_k: int, hidden: int, intermediate: int) -> float:
    token_expert_rows = tokens * top_k
    gemm1 = 2.0 * token_expert_rows * hidden * (2.0 * intermediate)
    gemm2 = 2.0 * token_expert_rows * intermediate * hidden
    if component == "gemm1":
        return gemm1
    if component == "gemm2":
        return gemm2
    if component in {"e2e", "profiler_sum"}:
        return gemm1 + gemm2
    return 0.0


def safe_tflops(flops: float, ms: float) -> float:
    return flops / (ms / 1000.0) / 1e12 if flops > 0 and ms > 0 else 0.0


def make_mode_data(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    backends: list[str] = []
    config: dict[str, Any] = {}
    runtime_by_backend: dict[str, dict[str, Any]] = {}

    for record in records:
        if record.get("mode") != mode or record.get("status") != "ok":
            continue
        report_path = record.get("output_json")
        if not report_path:
            continue
        report = load_json(Path(report_path))
        backend = str(report.get("moe_backend", record.get("backend", "")))
        if backend not in backends:
            backends.append(backend)
        config = report.get("config", config)
        runtime_by_backend[backend] = report.get("runtime", {})
        hidden = int(config.get("hidden_size", 0) or 0)
        intermediate = int(config.get("intermediate_size", 0) or 0)
        top_k = int(config.get("top_k", 1) or 1)
        case_tokens: dict[str, Any] = {}

        for result in report.get("results", []):
            tokens = int(result["tokens"])
            category_ms_by_component = component_ms(result)
            ranks = result.get("ranks", [])
            local_rows_by_rank = {
                str(rank.get("rank")): int(rank.get("local_token_expert_rows", 0) or 0)
                for rank in ranks
            }
            case_tokens[str(tokens)] = {
                "tokens": tokens,
                "event_ms_max": float(result.get("event_ms_max", 0.0) or 0.0),
                "global_counts": result.get("global_counts", []),
                "local_rows_by_rank": local_rows_by_rank,
                "rank_expert_ranges": [
                    {
                        "rank": rank.get("rank"),
                        "first_expert": (rank.get("loaded_experts") or [None])[0],
                        "last_expert": (rank.get("loaded_experts") or [None])[-1],
                        "local_token_expert_rows": rank.get("local_token_expert_rows", 0),
                    }
                    for rank in ranks
                ],
                "component_ms": category_ms_by_component,
            }
            for component, ms in category_ms_by_component.items():
                if ms <= 0:
                    continue
                flops = flops_for_component(component, tokens, top_k, hidden, intermediate)
                points.append(
                    {
                        "backend": backend,
                        "tokens": tokens,
                        "component": component,
                        "ms": ms,
                        "tokens_per_s": tokens / ms * 1000.0,
                        "tflops": safe_tflops(flops, ms),
                    }
                )

        cases[backend] = {
            "runtime": report.get("runtime", {}),
            "tokens": case_tokens,
        }

    backends.sort(key=lambda item: ["auto", "flashinfer_trtllm", "flashinfer_cutlass", "triton"].index(item) if item in ["auto", "flashinfer_trtllm", "flashinfer_cutlass", "triton"] else 99)
    return {
        "label": MODE_LABELS.get(mode, mode),
        "note": MODE_NOTES.get(mode, ""),
        "config": config,
        "backends": backends,
        "runtime_by_backend": runtime_by_backend,
        "points": points,
        "cases": cases,
    }


def build_report(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary.get("records", [])
    modes = [mode for mode in summary.get("modes", []) if any(r.get("mode") == mode for r in records)]
    mode_data = {mode: make_mode_data(mode, records) for mode in modes}
    failures = [
        {
            "mode": record.get("mode", ""),
            "backend": record.get("backend", ""),
            "returncode": record.get("returncode", ""),
            "error_tail": record.get("error_tail", "")[-1800:],
            "log": record.get("log", ""),
        }
        for record in records
        if record.get("status") != "ok"
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "model_path": summary.get("model_path", ""),
            "layer_id": summary.get("layer_id", ""),
            "expert_id": summary.get("expert_id", ""),
            "cuda_visible_devices": summary.get("cuda_visible_devices", ""),
            "all2all_backend": summary.get("all2all_backend", ""),
            "nproc_per_node": summary.get("nproc_per_node", ""),
        },
        "components": COMPONENTS,
        "modes": mode_data,
        "failures": failures,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HunyuanImage3 EP=2 MoE Backend Sweep</title>
<style>
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }
header { padding: 24px 28px 14px; background: #ffffff; border-bottom: 1px solid #e5e7eb; }
h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }
h2 { margin: 18px 0 10px; font-size: 18px; }
h3 { margin: 0 0 8px; font-size: 15px; }
.muted { color: #6b7280; font-size: 13px; line-height: 1.45; }
.wrap { padding: 18px 28px 28px; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
button, select, label.checkbox { border: 1px solid #cbd5e1; background: #ffffff; color: #111827; border-radius: 6px; padding: 7px 10px; font-size: 13px; }
button.active { background: #111827; color: #ffffff; border-color: #111827; }
.tab { display: none; }
.tab.active { display: block; }
.grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(360px, 0.95fr); gap: 14px; align-items: start; }
.panel { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
.backendChecks { display: flex; flex-wrap: wrap; gap: 8px; }
label.checkbox { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
.plot { width: 100%; height: 430px; display: block; background: #fbfdff; border: 1px solid #eef2f7; border-radius: 6px; }
.smallPlot { height: 300px; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: #ffffff; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 0; font-size: 12px; color: #374151; }
.swatch { width: 11px; height: 11px; display: inline-block; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
@media (max-width: 980px) { .grid { grid-template-columns: 1fr; } header, .wrap { padding-left: 16px; padding-right: 16px; } }
</style>
</head>
<body>
<header>
  <h1>HunyuanImage3 EP=2 MoE Backend Sweep</h1>
  <div class="muted" id="summaryLine"></div>
  <div class="muted">EP=2 here uses vLLM MoE expert parallel over a 2-rank TP group: `tp_size=1`, `ep_size=2`, each rank owns 32 experts. With `dp_size=1`, vLLM uses the no-DP/EP prepare/finalize path and the visible inter-rank cost is output cross-device reduce, not DeepEP all-to-all.</div>
</header>
<main class="wrap">
  <div class="tabs" id="tabs"></div>
  <div id="tabBodies"></div>
</main>
<script id="report-data" type="application/json">__REPORT_JSON__</script>
<script>
const report = JSON.parse(document.getElementById("report-data").textContent);
const metricLabels = {tokens_per_s: "input tokens/s", ms: "latency ms", tflops: "TFLOPs"};
const colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4b5563", "#65a30d"];
const state = {};
function fmt(v, metric) {
  if (!Number.isFinite(v)) return "";
  if (metric === "tokens_per_s") {
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return v.toFixed(1);
  }
  if (metric === "tflops") return v.toFixed(1);
  return v.toFixed(3);
}
function make(tag, attrs={}, text=null) {
  const el = document.createElementNS(tag === "svg" || tag === "path" || tag === "circle" || tag === "line" || tag === "rect" || tag === "text" || tag === "polyline" ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k, v));
  if (text !== null) el.textContent = text;
  return el;
}
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
function modeTokens(modeData) {
  return [...new Set(modeData.points.map(p => p.tokens))].sort((a,b) => a-b);
}
function selectedBackends(mode) {
  return [...document.querySelectorAll(`[data-mode="${mode}"][data-kind="backend"]:checked`)].map(el => el.value);
}
function nearestToken(x, tokens, ml, plotW) {
  if (!tokens.length) return null;
  const minLog = Math.log2(tokens[0]);
  const maxLog = Math.log2(tokens[tokens.length - 1]);
  const valueLog = minLog + Math.max(0, Math.min(1, (x - ml) / plotW)) * (maxLog - minLog || 1);
  return tokens.reduce((best, t) => Math.abs(Math.log2(t) - valueLog) < Math.abs(Math.log2(best) - valueLog) ? t : best, tokens[0]);
}
function renderLine(mode) {
  const modeData = report.modes[mode];
  const st = state[mode];
  st.backends = selectedBackends(mode);
  const svg = document.getElementById(`${mode}-line`);
  clear(svg);
  const w = svg.clientWidth || 900, h = svg.clientHeight || 430;
  const ml = 64, mr = 24, mt = 22, mb = 50;
  const plotW = w - ml - mr, plotH = h - mt - mb;
  const tokens = modeTokens(modeData);
  const points = modeData.points.filter(p => p.component === st.component && st.backends.includes(p.backend));
  if (!tokens.length || !points.length) {
    svg.appendChild(make("text", {x: 24, y: 48, fill: "#6b7280", "font-size": 14}, "no data"));
    return;
  }
  const minLog = Math.log2(tokens[0]), maxLog = Math.log2(tokens[tokens.length - 1]);
  const sx = t => ml + ((Math.log2(t) - minLog) / (maxLog - minLog || 1)) * plotW;
  const ymax = Math.max(...points.map(p => p[st.metric] || 0), 1e-9) * 1.08;
  const sy = v => mt + plotH - (v / ymax) * plotH;
  svg.appendChild(make("line", {x1: ml, y1: mt, x2: ml, y2: mt + plotH, stroke: "#94a3b8"}));
  svg.appendChild(make("line", {x1: ml, y1: mt + plotH, x2: ml + plotW, y2: mt + plotH, stroke: "#94a3b8"}));
  for (let i = 0; i <= 4; i++) {
    const yv = ymax * i / 4;
    const y = sy(yv);
    svg.appendChild(make("line", {x1: ml, y1: y, x2: ml + plotW, y2: y, stroke: "#e5e7eb"}));
    svg.appendChild(make("text", {x: 8, y: y + 4, fill: "#64748b", "font-size": 11}, fmt(yv, st.metric)));
  }
  tokens.forEach(t => {
    const x = sx(t);
    svg.appendChild(make("text", {x, y: mt + plotH + 22, fill: "#64748b", "font-size": 11, "text-anchor": "middle"}, String(t)));
  });
  const byBackend = new Map();
  points.forEach(p => {
    if (!byBackend.has(p.backend)) byBackend.set(p.backend, []);
    byBackend.get(p.backend).push(p);
  });
  [...byBackend.entries()].forEach(([backend, arr], i) => {
    arr.sort((a,b) => a.tokens - b.tokens);
    const d = arr.map(p => `${sx(p.tokens)},${sy(p[st.metric] || 0)}`).join(" ");
    svg.appendChild(make("polyline", {points: d, fill: "none", stroke: colors[i % colors.length], "stroke-width": 2}));
    arr.forEach(p => {
      const c = make("circle", {cx: sx(p.tokens), cy: sy(p[st.metric] || 0), r: 4, fill: colors[i % colors.length], "data-token": p.tokens, "data-backend": backend});
      c.addEventListener("click", () => { st.token = p.tokens; st.backend = backend; renderSide(mode); });
      svg.appendChild(c);
    });
  });
  const cursorX = sx(st.token || tokens[0]);
  svg.appendChild(make("line", {x1: cursorX, y1: mt, x2: cursorX, y2: mt + plotH, stroke: "#111827", "stroke-dasharray": "4 4"}));
  svg.addEventListener("mousemove", ev => {
    const rect = svg.getBoundingClientRect();
    const token = nearestToken(ev.clientX - rect.left, tokens, ml, plotW);
    if (token && token !== st.token) {
      st.token = token;
      renderLine(mode);
      renderSide(mode);
    }
  }, {once: true});
  renderLegend(mode, byBackend);
}
function renderLegend(mode, byBackend) {
  const div = document.getElementById(`${mode}-legend`);
  clear(div);
  [...byBackend.keys()].forEach((backend, i) => {
    const item = document.createElement("span");
    item.innerHTML = `<span class="swatch" style="background:${colors[i % colors.length]}"></span>${backend}`;
    div.appendChild(item);
  });
}
function renderBars(mode) {
  const modeData = report.modes[mode];
  const st = state[mode];
  const svg = document.getElementById(`${mode}-bars`);
  clear(svg);
  const w = svg.clientWidth || 700, h = svg.clientHeight || 300;
  const ml = 54, mr = 18, mt = 24, mb = 72;
  const comps = ["e2e", "gemm1", "gemm2", "comm", "routing", "finalize"];
  const rows = [];
  for (const backend of st.backends) {
    for (const comp of comps) {
      const p = modeData.points.find(x => x.backend === backend && x.tokens === st.token && x.component === comp);
      if (p) rows.push({backend, comp, value: p[st.metric] || 0});
    }
  }
  if (!rows.length) {
    svg.appendChild(make("text", {x: 20, y: 42, fill: "#6b7280", "font-size": 13}, "no component data"));
    return;
  }
  const ymax = Math.max(...rows.map(r => r.value), 1e-9) * 1.08;
  const plotW = w - ml - mr, plotH = h - mt - mb;
  const groupW = plotW / Math.max(st.backends.length, 1);
  const barW = Math.max(3, groupW / comps.length - 3);
  const sy = v => mt + plotH - (v / ymax) * plotH;
  svg.appendChild(make("line", {x1: ml, y1: mt, x2: ml, y2: mt + plotH, stroke: "#94a3b8"}));
  svg.appendChild(make("line", {x1: ml, y1: mt + plotH, x2: ml + plotW, y2: mt + plotH, stroke: "#94a3b8"}));
  st.backends.forEach((backend, bi) => {
    comps.forEach((comp, ci) => {
      const row = rows.find(r => r.backend === backend && r.comp === comp);
      if (!row) return;
      const x = ml + bi * groupW + ci * (barW + 3) + 5;
      const y = sy(row.value);
      svg.appendChild(make("rect", {x, y, width: barW, height: mt + plotH - y, fill: colors[ci % colors.length]}));
    });
    const xLabel = ml + bi * groupW + groupW / 2;
    svg.appendChild(make("text", {x: xLabel, y: mt + plotH + 20, fill: "#475569", "font-size": 10, "text-anchor": "middle"}, backend));
  });
  for (let i = 0; i <= 4; i++) {
    const yv = ymax * i / 4;
    const y = sy(yv);
    svg.appendChild(make("line", {x1: ml, y1: y, x2: ml + plotW, y2: y, stroke: "#e5e7eb"}));
    svg.appendChild(make("text", {x: 6, y: y + 4, fill: "#64748b", "font-size": 10}, fmt(yv, st.metric)));
  }
}
function renderExpert(mode) {
  const modeData = report.modes[mode];
  const st = state[mode];
  const backend = st.backend && modeData.cases[st.backend] ? st.backend : st.backends[0];
  st.backend = backend;
  const tokenCase = modeData.cases[backend]?.tokens?.[String(st.token)];
  const svg = document.getElementById(`${mode}-experts`);
  clear(svg);
  if (!tokenCase) {
    svg.appendChild(make("text", {x: 20, y: 42, fill: "#6b7280", "font-size": 13}, "no expert distribution"));
    return;
  }
  const counts = tokenCase.global_counts || [];
  const w = svg.clientWidth || 700, h = svg.clientHeight || 300;
  const ml = 44, mr = 18, mt = 22, mb = 42;
  const plotW = w - ml - mr, plotH = h - mt - mb;
  const ymax = Math.max(...counts, 1);
  const barW = plotW / counts.length;
  const sy = v => mt + plotH - (v / ymax) * plotH;
  svg.appendChild(make("line", {x1: ml, y1: mt, x2: ml, y2: mt + plotH, stroke: "#94a3b8"}));
  svg.appendChild(make("line", {x1: ml, y1: mt + plotH, x2: ml + plotW, y2: mt + plotH, stroke: "#94a3b8"}));
  counts.forEach((v, i) => {
    const x = ml + i * barW;
    const y = sy(v);
    const fill = i < 32 ? "#2563eb" : "#059669";
    svg.appendChild(make("rect", {x, y, width: Math.max(1, barW - 1), height: mt + plotH - y, fill}));
  });
  [0, 16, 32, 48, 63].forEach(i => {
    const x = ml + i * barW + barW / 2;
    svg.appendChild(make("text", {x, y: mt + plotH + 18, fill: "#64748b", "font-size": 10, "text-anchor": "middle"}, String(i)));
  });
  svg.appendChild(make("text", {x: 5, y: sy(ymax) + 4, fill: "#64748b", "font-size": 10}, String(ymax)));
  const details = document.getElementById(`${mode}-details`);
  const runtime = modeData.cases[backend].runtime || {};
  details.textContent =
    `selected: backend=${backend}, tokens=${st.token}\n` +
    `effective_backend=${runtime.effective_moe_backend || ""}, prepare_finalize=${runtime.prepare_finalize || ""}, experts=${runtime.experts || ""}\n` +
    `component_ms=${JSON.stringify(tokenCase.component_ms)}\n` +
    `local_rows_by_rank=${JSON.stringify(tokenCase.local_rows_by_rank)}\n` +
    `rank_expert_ranges=${JSON.stringify(tokenCase.rank_expert_ranges)}`;
}
function renderSide(mode) {
  document.getElementById(`${mode}-title`).textContent = `${report.modes[mode].label}: token=${state[mode].token}`;
  renderBars(mode);
  renderExpert(mode);
}
function renderMode(mode) {
  renderLine(mode);
  renderSide(mode);
}
function setupMode(mode) {
  const modeData = report.modes[mode];
  const tokens = modeTokens(modeData);
  state[mode] = {metric: "tokens_per_s", component: "e2e", token: tokens[0], backend: modeData.backends[0], backends: [...modeData.backends]};
  const body = document.createElement("section");
  body.id = `tab-${mode}`;
  body.className = "tab";
  body.innerHTML = `
    <div class="panel" style="margin-bottom:14px">
      <h2>${modeData.label}</h2>
      <div class="muted">${modeData.note}</div>
      <div class="controls">
        <label>Metric <select id="${mode}-metric">
          <option value="tokens_per_s">input tokens/s</option>
          <option value="ms">latency ms</option>
          <option value="tflops">TFLOPs</option>
        </select></label>
        <label>Component <select id="${mode}-component">
          ${report.components.map(c => `<option value="${c.key}">${c.label}</option>`).join("")}
        </select></label>
        <div class="backendChecks">
          ${modeData.backends.map(b => `<label class="checkbox"><input type="checkbox" data-mode="${mode}" data-kind="backend" value="${b}" checked>${b}</label>`).join("")}
        </div>
      </div>
    </div>
    <div class="grid">
      <div class="panel">
        <h3>${modeData.label} backend curves</h3>
        <svg id="${mode}-line" class="plot"></svg>
        <div id="${mode}-legend" class="legend"></div>
      </div>
      <div class="panel">
        <h3 id="${mode}-title"></h3>
        <svg id="${mode}-bars" class="plot smallPlot"></svg>
        <h3 style="margin-top:14px">Expert token distribution</h3>
        <svg id="${mode}-experts" class="plot smallPlot"></svg>
        <pre id="${mode}-details" class="muted"></pre>
      </div>
    </div>`;
  document.getElementById("tabBodies").appendChild(body);
  document.getElementById(`${mode}-metric`).addEventListener("change", ev => { state[mode].metric = ev.target.value; renderMode(mode); });
  document.getElementById(`${mode}-component`).addEventListener("change", ev => { state[mode].component = ev.target.value; renderLine(mode); });
  document.querySelectorAll(`[data-mode="${mode}"][data-kind="backend"]`).forEach(el => el.addEventListener("change", () => renderMode(mode)));
}
function setupFailures() {
  const body = document.createElement("section");
  body.id = "tab-failures";
  body.className = "tab";
  const rows = report.failures.map(f => `<tr><td>${f.mode}</td><td>${f.backend}</td><td>${f.returncode}</td><td><pre class="muted">${String(f.error_tail || "").replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</pre></td></tr>`).join("");
  body.innerHTML = `<div class="panel"><h2>Backend failures</h2><table><thead><tr><th>mode</th><th>backend</th><th>rc</th><th>tail</th></tr></thead><tbody>${rows || '<tr><td colspan="4">none</td></tr>'}</tbody></table></div>`;
  document.getElementById("tabBodies").appendChild(body);
}
function activate(tabId) {
  document.querySelectorAll(".tabBtn").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabId));
  document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.id === tabId));
}
document.getElementById("summaryLine").textContent =
  `model=${report.summary.model_path}, layer=${report.summary.layer_id}, expert=${report.summary.expert_id}, GPUs=${report.summary.cuda_visible_devices}, generated=${report.generated_at}`;
for (const mode of Object.keys(report.modes)) {
  setupMode(mode);
  const btn = document.createElement("button");
  btn.className = "tabBtn";
  btn.dataset.tab = `tab-${mode}`;
  btn.textContent = report.modes[mode].label;
  btn.addEventListener("click", () => activate(btn.dataset.tab));
  document.getElementById("tabs").appendChild(btn);
}
setupFailures();
const failBtn = document.createElement("button");
failBtn.className = "tabBtn";
failBtn.dataset.tab = "tab-failures";
failBtn.textContent = `Failures (${report.failures.length})`;
failBtn.addEventListener("click", () => activate("tab-failures"));
document.getElementById("tabs").appendChild(failBtn);
const firstTab = document.querySelector(".tabBtn");
if (firstTab) activate(firstTab.dataset.tab);
for (const mode of Object.keys(report.modes)) renderMode(mode);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    summary = load_json(Path(args.summary_json))
    report = build_report(summary)
    html = HTML_TEMPLATE.replace("__REPORT_JSON__", json.dumps(report, ensure_ascii=False))
    output = Path(args.output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
