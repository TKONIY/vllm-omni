#!/usr/bin/env bash
# HunyuanImage3 legacy stage config: image+text-to-image with AR->DiT KV reuse.
#
# What this launches:
# - `hunyuan_image3_it2i_kv_reuse.yaml`
# - two-stage AR + DiT pipeline;
# - AR stage on GPUs 0-1, DiT/VAE stage on GPUs 2-3;
# - DiT uses tensor parallel size 2 with expert parallel enabled;
# - AR sends KV cache to DiT through the configured RDMA connector;
# - useful for validating native multiturn/context reuse across AR and DiT.
#
# Use this when debugging the compact 4-GPU AR+DiT KV-reuse path.

set -Eeuo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
HUNYUAN_STAGE_CONFIG="${HUNYUAN_STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/hunyuan_image3_it2i_kv_reuse.yaml}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hunyuan_image3_debugpy_common.sh"

hunyuan_run_vllm_omni_debugpy \
  --stage-configs-path "${HUNYUAN_STAGE_CONFIG}" \
  "$@"
