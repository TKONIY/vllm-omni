#!/usr/bin/env bash
# Launch one HunyuanImage3 stage under debugpy.
#
# Examples:
#   HUNYUAN_STAGE_CONFIG=vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml \
#   STAGE_ID=0 DEBUGPY_PORT=5678 bash docs/uad/hunyuan_image3_stage_process_debugpy.sh
#
#   HUNYUAN_STAGE_CONFIG=vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml \
#   STAGE_ID=1 DEBUGPY_PORT=5679 bash docs/uad/hunyuan_image3_stage_process_debugpy.sh

set -Eeuo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
STAGE_ID="${STAGE_ID:-0}"
DEBUGPY_PORT="${DEBUGPY_PORT:-$((5678 + STAGE_ID))}"
HUNYUAN_STAGE_CONFIG="${HUNYUAN_STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml}"
OMNI_MASTER_ADDRESS="${OMNI_MASTER_ADDRESS:-127.0.0.1}"
OMNI_MASTER_PORT="${OMNI_MASTER_PORT:-26000}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hunyuan_image3_debugpy_common.sh"

stage_args=(
  --stage-configs-path "${HUNYUAN_STAGE_CONFIG}"
  --stage-id "${STAGE_ID}"
  --omni-master-address "${OMNI_MASTER_ADDRESS}"
  --omni-master-port "${OMNI_MASTER_PORT}"
)

HEADLESS="${HEADLESS:-}"
if [[ -z "${HEADLESS}" ]]; then
  if (( STAGE_ID == 0 )); then
    HEADLESS=0
  else
    HEADLESS=1
  fi
fi

if hunyuan_bool_true "${HEADLESS}"; then
  stage_args+=(--headless)
fi

hunyuan_run_vllm_omni_debugpy "${stage_args[@]}" "$@"
