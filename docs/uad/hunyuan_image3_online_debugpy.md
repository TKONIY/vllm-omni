# HunyuanImage3 Online Debugpy Launchers

These scripts start HunyuanImage3 online serving through `debugpy` so VS Code
can attach before server startup. They run from the repository root and set
`PYTHONPATH` to the working tree.

## Common Environment

All scripts accept the same environment overrides:

```bash
MODEL=tencent/HunyuanImage-3.0-Instruct
HOST=0.0.0.0
PORT=8091
DEBUGPY_HOST=0.0.0.0
DEBUGPY_PORT=5678
DEBUGPY_WAIT_FOR_CLIENT=1
PYTHON_BIN=python
```

Set `DEBUGPY_WAIT_FOR_CLIENT=0` if you want the server to start immediately.
Any extra script arguments are appended to the underlying `vllm-omni serve`
command.

## DiT-Only Recipe Scripts

| Deployment | Script |
| --- | --- |
| TP=4 + FP8 | `docs/uad/hunyuan_image3_dit_tp4_fp8_debugpy.sh` |
| TP=2 + FP8 + Ulysses SP=2 | `docs/uad/hunyuan_image3_dit_tp2_fp8_sp2_debugpy.sh` |
| TP=2 + FP8 + CFG parallel=2 | `docs/uad/hunyuan_image3_dit_tp2_fp8_cfgp2_debugpy.sh` |

Example:

```bash
bash docs/uad/hunyuan_image3_dit_tp2_fp8_sp2_debugpy.sh
```

## Stage-Config Scripts

| Deployment config | Script |
| --- | --- |
| `hunyuan_image3_t2i.yaml` | `docs/uad/hunyuan_image3_stage_t2i_debugpy.sh` |
| `hunyuan_image3_t2i_2gpu.yaml` | `docs/uad/hunyuan_image3_stage_t2i_2gpu_debugpy.sh` |
| `hunyuan_image3_i2t.yaml` | `docs/uad/hunyuan_image3_stage_i2t_debugpy.sh` |
| `hunyuan_image3_t2t.yaml` | `docs/uad/hunyuan_image3_stage_t2t_debugpy.sh` |
| `hunyuan_image3_it2i.yaml` | `docs/uad/hunyuan_image3_stage_it2i_debugpy.sh` |
| `hunyuan_image3_it2i_kv_reuse.yaml` | `docs/uad/hunyuan_image3_stage_it2i_kv_reuse_debugpy.sh` |
| `hunyuan_image3_moe.yaml` | `docs/uad/hunyuan_image3_stage_moe_debugpy.sh` |
| `hunyuan_image3_moe_dit_2gpu_fp8.yaml` | `docs/uad/hunyuan_image3_stage_moe_dit_2gpu_fp8_debugpy.sh` |

Example:

```bash
bash docs/uad/hunyuan_image3_stage_moe_debugpy.sh
```

## Split-Stage Debugging

Use `hunyuan_image3_stage_process_debugpy.sh` when you want each stage to have
its own debugpy port. Start stage 0 first, then the headless downstream stages.

```bash
HUNYUAN_STAGE_CONFIG=vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml \
STAGE_ID=0 DEBUGPY_PORT=5678 \
bash docs/uad/hunyuan_image3_stage_process_debugpy.sh

HUNYUAN_STAGE_CONFIG=vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml \
STAGE_ID=1 DEBUGPY_PORT=5679 \
bash docs/uad/hunyuan_image3_stage_process_debugpy.sh
```

Use the matching VS Code attach configuration for the selected port.
