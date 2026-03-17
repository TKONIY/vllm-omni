# cuda/ 子模块索引

## 模块概述

`cuda/` 子模块实现了基于 NVIDIA CUDA/GPU 的 OmniPlatform。CUDA 平台是 vllm-omni 的默认平台，直接复用 vLLM 原有的 GPU Worker 和 ModelRunner，无需额外的平台专用 Worker 实现。

## 文件列表

| 文件 | 说明 |
|------|------|
| [__init__.py.md](./__init__.py.md) | 包导出 |
| [platform.py.md](./platform.py.md) | CudaOmniPlatform 实现 |

## 架构特点

- CUDA 平台的 Worker 类指向通用 GPU 实现（`vllm_omni.worker.gpu_ar_worker.GPUARWorker`），不需要平台专用 Worker。
- 阶段配置使用默认路径 `vllm_omni/model_executor/stage_configs`。
- 支持 Flash Attention（计算能力 >= 8.0 且 < 10.0）和 `torch.compile` inductor 后端。
