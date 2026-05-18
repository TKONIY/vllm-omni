#!/usr/bin/env bash
# Orchestrate the four PD-disaggregated Qwen3-Omni stages, each in its
# own top-level Python process so each one can be wrapped in its own
# debugpy listener.
#
# Default debugpy port layout (matches .vscode/launch.json):
#   stage 0 (prefill + API)  -> 5678
#   stage 1 (decode)         -> 5679
#   stage 2 (talker)         -> 5680
#   stage 3 (code2wav)       -> 5681
#
# Usage:
#   ./run_all_stages_disagg.sh                  # no debug, all four stages
#   ./run_all_stages_disagg.sh --debug          # all four stages wait for VSCode
#   ./run_all_stages_disagg.sh --debug --debug-stages 0,1
#                                               # only stages 0 and 1 wait, others run free
#
# Behaviour:
#   * Stages 1, 2, 3 run in the background; their stdout/stderr stream to
#     /tmp/vllm_omni_disagg_stage<N>.log.
#   * Stage 0 runs in the foreground; Ctrl-C tears down all four.
#   * In --debug mode, every selected stage uses wait-for-client. Attach
#     them via the "Compound: Attach all PD-disagg stages" launch in
#     VSCode (or attach each port individually).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_server_disagg.sh"

MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
HOST="0.0.0.0"
STAGE_CONFIGS="${SCRIPT_DIR}/qwen3_omni_pd_disagg.yaml"
MASTER_ADDR="127.0.0.1"
MASTER_PORT=26000

DEBUG=false
DEBUG_STAGES="0,1,2,3"
DEBUG_HOST="0.0.0.0"
# stage_id -> port
DEBUG_PORTS=(5678 5679 5680 5681)

LOG_DIR="${VLLM_OMNI_DISAGG_LOG_DIR:-/tmp}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

  --model MODEL           Model id/path (default: ${MODEL})
  --port PORT             API server port for stage 0 (default: ${PORT})
  --host HOST             API server bind host (default: ${HOST})
  --stage-configs PATH    PD-disagg stage configs YAML (default: ${STAGE_CONFIGS})
  --master-addr HOST      Omni orchestrator host (default: ${MASTER_ADDR})
  --master-port PORT      Omni orchestrator port (default: ${MASTER_PORT})

  --debug                 Wrap every selected stage in debugpy wait-for-client
  --debug-stages LIST     Comma-separated stage ids to debug-wrap (default: ${DEBUG_STAGES})
  --debug-host HOST       debugpy listen host (default: ${DEBUG_HOST})
  --debug-ports LIST      Comma-separated debugpy ports for stages 0..3
                          (default: ${DEBUG_PORTS[*]})

  --log-dir DIR           Directory for background stage logs (default: ${LOG_DIR})

  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";         shift 2 ;;
        --port)          PORT="$2";          shift 2 ;;
        --host)          HOST="$2";          shift 2 ;;
        --stage-configs) STAGE_CONFIGS="$2"; shift 2 ;;
        --master-addr)   MASTER_ADDR="$2";   shift 2 ;;
        --master-port)   MASTER_PORT="$2";   shift 2 ;;
        --debug)         DEBUG=true;         shift   ;;
        --debug-stages)  DEBUG_STAGES="$2";  shift 2 ;;
        --debug-host)    DEBUG_HOST="$2";    shift 2 ;;
        --debug-ports)
            IFS=',' read -r -a DEBUG_PORTS <<< "$2"
            if [[ ${#DEBUG_PORTS[@]} -ne 4 ]]; then
                echo "Error: --debug-ports needs 4 comma-separated values (stages 0..3)" >&2
                exit 1
            fi
            shift 2 ;;
        --log-dir)       LOG_DIR="$2";       shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "$LAUNCHER" ]]; then
    echo "Error: per-stage launcher not executable: $LAUNCHER" >&2
    exit 1
fi
if [[ ! -f "$STAGE_CONFIGS" ]]; then
    echo "Error: stage configs YAML not found: $STAGE_CONFIGS" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

is_debug_stage() {
    local sid="$1"
    IFS=',' read -r -a _ds <<< "$DEBUG_STAGES"
    for s in "${_ds[@]}"; do
        if [[ "$s" == "$sid" ]]; then return 0; fi
    done
    return 1
}

declare -a CHILD_PIDS=()

cleanup() {
    echo ""
    echo "Shutting down stages 1..3..."
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "All stages stopped."
}
trap cleanup SIGINT SIGTERM EXIT

# Common args every stage shares
COMMON_ARGS=(
    --model "$MODEL"
    --stage-configs "$STAGE_CONFIGS"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
)

build_stage_cmd() {
    local sid="$1"
    local -n out_ref="$2"
    out_ref=("$LAUNCHER" "${COMMON_ARGS[@]}" --stage-id "$sid")
    if [[ "$sid" == "0" ]]; then
        out_ref+=(--host "$HOST" --port "$PORT")
    fi
    if [[ "$DEBUG" == "true" ]] && is_debug_stage "$sid"; then
        out_ref+=(--debug --debug-host "$DEBUG_HOST" --debug-port "${DEBUG_PORTS[$sid]}")
    fi
}

echo "=========================================="
echo "vLLM-Omni PD-Disaggregated — 4-stage launch"
echo "=========================================="
echo "Model:         $MODEL"
echo "API server:    http://${HOST}:${PORT}"
echo "Orchestrator:  ${MASTER_ADDR}:${MASTER_PORT}"
echo "Stage configs: $STAGE_CONFIGS"
echo "Logs dir:      $LOG_DIR"
if [[ "$DEBUG" == "true" ]]; then
    echo "Debug stages:  $DEBUG_STAGES"
    echo "Debug ports:   ${DEBUG_PORTS[*]}  (stage 0..3)"
fi
echo "=========================================="

# Start stages 1, 2, 3 in the background first so stage 0 (which holds
# the API socket and prints to the foreground) is the last to come up.
for sid in 1 2 3; do
    cmd=()
    build_stage_cmd "$sid" cmd
    log="${LOG_DIR}/vllm_omni_disagg_stage${sid}.log"
    echo ""
    echo "[stage ${sid}] log -> ${log}"
    echo "[stage ${sid}] cmd: ${cmd[*]}"
    "${cmd[@]}" >"$log" 2>&1 &
    CHILD_PIDS+=($!)
done

# Stage 0 in foreground so its logs are visible and Ctrl-C goes through trap.
cmd=()
build_stage_cmd 0 cmd
echo ""
echo "[stage 0] cmd: ${cmd[*]}"
echo ""
"${cmd[@]}"
