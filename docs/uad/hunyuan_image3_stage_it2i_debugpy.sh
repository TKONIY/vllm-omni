#!/usr/bin/env bash
# HunyuanImage3 legacy stage config: image+text-to-image editing.
#
# What this launches:
# - `hunyuan_image3_it2i.yaml`
# - two-stage AR + DiT pipeline;
# - AR stage on GPUs 0-3, DiT/VAE stage on GPUs 4-7;
# - DiT uses tensor parallel size 4 with expert parallel enabled;
# - AR output is converted to DiT input through the stage input processor;
# - no explicit AR->DiT KV cache reuse connector.
#
# Use this when debugging the basic image+text-to-image staged pipeline.

set -Eeuo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
HUNYUAN_STAGE_CONFIG="${HUNYUAN_STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/hunyuan_image3_it2i.yaml}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hunyuan_image3_debugpy_common.sh"

hunyuan_run_vllm_omni_debugpy \
  --stage-configs-path "${HUNYUAN_STAGE_CONFIG}" \
  "$@"
