#!/usr/bin/env bash
# HunyuanImage3 legacy stage config: AR -> DiT KV-reuse MoE deployment.
#
# What this launches:
# - `hunyuan_image3_moe.yaml`
# - two-stage AR + DiT pipeline;
# - AR stage on GPUs 0-3, DiT/VAE stage on GPUs 4-7;
# - AR sends reusable cache after prefill; DiT receives it before denoising;
# - DiT/VAE stage runs on 4 GPUs with tensor parallel size 4;
# - expert/sequence/CFG parallel are disabled in this config.
#
# Use this as the default full-size AR+DiT debug configuration.

set -Eeuo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
HUNYUAN_STAGE_CONFIG="${HUNYUAN_STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hunyuan_image3_debugpy_common.sh"

hunyuan_run_vllm_omni_debugpy \
  --stage-configs-path "${HUNYUAN_STAGE_CONFIG}" \
  "$@"
