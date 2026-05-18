#!/usr/bin/env bash
# Launch ONE process of the Qwen3-Omni PD-disaggregated pipeline.
#
# Two launch shapes are supported:
#
# (1) Single-process orchestrator (default — same as before):
#     The whole 4-stage pipeline is spawned as mp.Process children of one
#     `vllm serve` parent. Only the API server (and its asyncio scheduler /
#     orchestrator) is debug-attachable through `--debug`. Stage / TP worker
#     subprocesses cannot be reached by this debugpy.
#
# (2) Stage-based CLI (`--stage-id N`):
#     Launch a single stage in its own top-level Python process. Stage 0
#     also hosts the API server; stages > 0 require `--headless`. Pair this
#     mode with `run_all_stages_disagg.sh` (or four terminals) so the four
#     stages can each get their own debugpy port and be attached
#     independently from VSCode.
#
# Usage:
#   ./run_server_disagg.sh [--model MODEL] [--port PORT] [--host HOST]
#                          [--stage-configs PATH]
#                          [--stage-id N] [--master-addr HOST] [--master-port PORT]
#                          [--debug] [--debug-port PORT] [--debug-host HOST]
#
# Examples:
#   # Single-process orchestrator
#   ./run_server_disagg.sh
#   ./run_server_disagg.sh --debug                # attach API server only
#
#   # Stage-based, one terminal per stage:
#   ./run_server_disagg.sh --stage-id 0 --master-addr 127.0.0.1 --master-port 26000 --debug --debug-port 5678
#   ./run_server_disagg.sh --stage-id 1 --master-addr 127.0.0.1 --master-port 26000 --debug --debug-port 5679
#   ./run_server_disagg.sh --stage-id 2 --master-addr 127.0.0.1 --master-port 26000 --debug --debug-port 5680
#   ./run_server_disagg.sh --stage-id 3 --master-addr 127.0.0.1 --master-port 26000 --debug --debug-port 5681

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
HOST="0.0.0.0"
STAGE_CONFIGS="${SCRIPT_DIR}/qwen3_omni_a5000.yaml"

STAGE_ID=""
MASTER_ADDR="127.0.0.1"
MASTER_PORT=26000

DEBUG=false
DEBUG_PORT=5678
DEBUG_HOST="0.0.0.0"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

  --model MODEL          Model id/path (default: ${MODEL})
  --port PORT            HTTP API port — only meaningful for stage 0 (default: ${PORT})
  --host HOST            Bind host for API server (default: ${HOST})
  --stage-configs PATH   PD-disagg stage configs YAML (default: ${STAGE_CONFIGS})

  --stage-id N           Launch a single stage (0..3) using the stage-based CLI.
                         Omit to keep the legacy single-process orchestrator.
  --master-addr HOST     Omni orchestrator host (default: ${MASTER_ADDR})
  --master-port PORT     Omni orchestrator port (default: ${MASTER_PORT})

  --debug                Wrap this process in debugpy with listen + wait-for-client
  --debug-port PORT      debugpy listen port (default: ${DEBUG_PORT})
  --debug-host HOST      debugpy listen host (default: ${DEBUG_HOST})

  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";         shift 2 ;;
        --port)          PORT="$2";          shift 2 ;;
        --host)          HOST="$2";          shift 2 ;;
        --stage-configs) STAGE_CONFIGS="$2"; shift 2 ;;
        --stage-id)      STAGE_ID="$2";      shift 2 ;;
        --master-addr)   MASTER_ADDR="$2";   shift 2 ;;
        --master-port)   MASTER_PORT="$2";   shift 2 ;;
        --debug)         DEBUG=true;         shift   ;;
        --debug-port)    DEBUG_PORT="$2";    shift 2 ;;
        --debug-host)    DEBUG_HOST="$2";    shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$STAGE_CONFIGS" ]]; then
    echo "Error: stage configs YAML not found: $STAGE_CONFIGS" >&2
    exit 1
fi

# Build the serve argv based on stage-id presence.
SERVE_ARGS=(serve "$MODEL" --omni --stage-configs-path "$STAGE_CONFIGS")

if [[ -n "$STAGE_ID" ]]; then
    SERVE_ARGS+=(
        --stage-id "$STAGE_ID"
        --omni-master-address "$MASTER_ADDR"
        --omni-master-port "$MASTER_PORT"
    )
    if [[ "$STAGE_ID" == "0" ]]; then
        SERVE_ARGS+=(--host "$HOST" --port "$PORT")
        ROLE="stage 0 (API server + thinker prefill)"
    else
        SERVE_ARGS+=(--headless)
        ROLE="stage ${STAGE_ID} (headless)"
    fi
else
    SERVE_ARGS+=(--host "$HOST" --port "$PORT")
    ROLE="single-process orchestrator"
fi

echo "=========================================="
echo "vLLM-Omni PD-Disaggregated — ${ROLE}"
echo "=========================================="
echo "Model:         $MODEL"
if [[ -z "$STAGE_ID" || "$STAGE_ID" == "0" ]]; then
    echo "API server:    http://${HOST}:${PORT}"
fi
if [[ -n "$STAGE_ID" ]]; then
    echo "Orchestrator:  ${MASTER_ADDR}:${MASTER_PORT}"
fi
echo "Stage configs: $STAGE_CONFIGS"
if [[ "$DEBUG" == "true" ]]; then
    echo "Debugpy:       ${DEBUG_HOST}:${DEBUG_PORT} (wait-for-client)"
fi
echo "=========================================="

if [[ "$DEBUG" == "true" ]]; then
    if ! python -c "import debugpy" >/dev/null 2>&1; then
        echo "Error: debugpy is not installed. Run: pip install debugpy" >&2
        exit 1
    fi
    VLLM_BIN="$(command -v vllm || true)"
    if [[ -z "$VLLM_BIN" ]]; then
        echo "Error: 'vllm' executable not found on PATH." >&2
        exit 1
    fi
    echo "Waiting for VSCode debugger to attach on ${DEBUG_HOST}:${DEBUG_PORT}..."
    exec python -Xfrozen_modules=off -m debugpy \
        --listen "${DEBUG_HOST}:${DEBUG_PORT}" \
        --wait-for-client \
        "$VLLM_BIN" "${SERVE_ARGS[@]}"
else
    exec vllm "${SERVE_ARGS[@]}"
fi
