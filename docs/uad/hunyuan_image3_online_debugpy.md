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

## AR + DiT Stage-Config Scripts

| Script | Config | Main Difference |
| --- | --- | --- |
| `docs/uad/hunyuan_image3_stage_it2i_debugpy.sh` | `hunyuan_image3_it2i.yaml` | Basic image+text-to-image staged pipeline. AR uses GPUs 0-3, DiT/VAE uses GPUs 4-7 with TP=4 and EP enabled. No explicit AR->DiT KV reuse connector. |
| `docs/uad/hunyuan_image3_stage_it2i_kv_reuse_debugpy.sh` | `hunyuan_image3_it2i_kv_reuse.yaml` | Compact 4-GPU AR+DiT path. AR uses GPUs 0-1, DiT/VAE uses GPUs 2-3 with TP=2 and EP enabled. AR sends KV cache to DiT through the configured RDMA connector. |
| `docs/uad/hunyuan_image3_stage_moe_debugpy.sh` | `hunyuan_image3_moe.yaml` | Default full-size AR+DiT KV-reuse path. AR uses GPUs 0-3, DiT/VAE uses GPUs 4-7 with TP=4. Expert, sequence, and CFG parallel are disabled in this config. |

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
