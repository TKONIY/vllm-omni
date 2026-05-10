#!/usr/bin/env bash
# Shared debugpy launcher for HunyuanImage3 online serving scripts.
#
# This file is not a deployment recipe by itself. Recipe scripts source it to:
# - run from the repository root with PYTHONPATH pointing at the working tree;
# - start `vllm_omni.entrypoints.cli.main serve ... --omni` under debugpy;
# - optionally block on VS Code attach before server startup.

set -Eeuo pipefail

HUNYUAN_DEBUGPY_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HUNYUAN_REPO_ROOT="$(cd -- "${HUNYUAN_DEBUGPY_SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_OMNI_CLI_MODULE="${VLLM_OMNI_CLI_MODULE:-vllm_omni.entrypoints.cli.main}"

HUNYUAN_MODEL="${MODEL:-${HUNYUAN_MODEL:-tencent/HunyuanImage-3.0-Instruct}}"
HUNYUAN_HOST="${HOST:-${HUNYUAN_HOST:-0.0.0.0}}"
HUNYUAN_PORT="${PORT:-${HUNYUAN_PORT:-8091}}"

DEBUGPY_HOST="${DEBUGPY_HOST:-0.0.0.0}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
DEBUGPY_WAIT_FOR_CLIENT="${DEBUGPY_WAIT_FOR_CLIENT:-1}"

export PYTHONPATH="${HUNYUAN_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYDEVD_DISABLE_FILE_VALIDATION="${PYDEVD_DISABLE_FILE_VALIDATION:-1}"
export VLLM_LOGGING_COLOR="${VLLM_LOGGING_COLOR:-1}"

hunyuan_bool_true() {
  case "${1,,}" in
    1 | true | yes | y | on) return 0 ;;
    *) return 1 ;;
  esac
}

hunyuan_print_command() {
  printf '\n[debugpy] repo: %s\n' "${HUNYUAN_REPO_ROOT}"
  printf '[debugpy] listen: %s:%s\n' "${DEBUGPY_HOST}" "${DEBUGPY_PORT}"
  if hunyuan_bool_true "${DEBUGPY_WAIT_FOR_CLIENT}"; then
    printf '[debugpy] waiting for VS Code attach before server startup\n'
  fi
  printf '[debugpy] command:'
  printf ' %q' "$@"
  printf '\n\n'
}

hunyuan_run_vllm_omni_debugpy() {
  cd "${HUNYUAN_REPO_ROOT}"

  local debugpy_args=(
    "${PYTHON_BIN}" "-m" "debugpy"
    "--listen" "${DEBUGPY_HOST}:${DEBUGPY_PORT}"
  )
  if hunyuan_bool_true "${DEBUGPY_WAIT_FOR_CLIENT}"; then
    debugpy_args+=("--wait-for-client")
  fi

  local cmd=(
    "${debugpy_args[@]}"
    "-m" "${VLLM_OMNI_CLI_MODULE}"
    "serve" "${HUNYUAN_MODEL}"
    "--omni"
    "--host" "${HUNYUAN_HOST}"
    "--port" "${HUNYUAN_PORT}"
  )
  cmd+=("$@")

  hunyuan_print_command "${cmd[@]}"
  exec "${cmd[@]}"
}
