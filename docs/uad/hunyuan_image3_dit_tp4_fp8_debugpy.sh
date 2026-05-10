#!/usr/bin/env bash
# DiT-only HunyuanImage3 online serving: TP=4 + FP8.

set -Eeuo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hunyuan_image3_debugpy_common.sh"

hunyuan_run_vllm_omni_debugpy \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler \
  "$@"
