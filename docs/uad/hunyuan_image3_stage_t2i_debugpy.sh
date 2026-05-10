#!/usr/bin/env bash
# HunyuanImage3 legacy stage config: DiT text-to-image/default 4-GPU stage.

set -Eeuo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
HUNYUAN_STAGE_CONFIG="${HUNYUAN_STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/hunyuan_image3_t2i.yaml}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hunyuan_image3_debugpy_common.sh"

hunyuan_run_vllm_omni_debugpy \
  --stage-configs-path "${HUNYUAN_STAGE_CONFIG}" \
  "$@"
