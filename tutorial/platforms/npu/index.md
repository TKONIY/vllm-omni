# npu/ 子模块索引

## 模块概述

`npu/` 子模块实现了基于华为 Ascend NPU 的 OmniPlatform。与 CUDA/ROCm 平台不同，NPU 平台由于硬件架构差异较大，需要完整的平台专用 Worker 和 ModelRunner 实现，是四个平台中代码量最大、定制化程度最高的子模块。

## 目录结构

```
npu/
├── __init__.py           # 包导出
├── platform.py           # NPUOmniPlatform 实现
├── models/               # NPU 专用模型算子
│   ├── __init__.py
│   └── hunyuan_fused_moe.py  # HunyuanFusedMoE Ascend 实现
├── worker/               # NPU 专用 Worker 和 ModelRunner
│   ├── __init__.py
│   ├── npu_model_runner.py           # NPU 基础 ModelRunner
│   ├── npu_ar_model_runner.py        # NPU 自回归 ModelRunner
│   ├── npu_ar_worker.py              # NPU 自回归 Worker
│   ├── npu_generation_model_runner.py # NPU 生成 ModelRunner
│   └── npu_generation_worker.py       # NPU 生成 Worker
└── stage_configs/        # NPU 阶段配置
    ├── qwen2_5_omni.yaml
    ├── qwen3_omni_moe.yaml
    ├── qwen3_omni_moe_async_chunk.yaml
    └── qwen3_tts.yaml
```

## 教程文件列表

| 文件 | 说明 |
|------|------|
| [__init__.py.md](./__init__.py.md) | 包导出 |
| [platform.py.md](./platform.py.md) | NPUOmniPlatform 实现 |
| [models/index.md](./models/index.md) | 模型算子索引 |
| [models/hunyuan_fused_moe.py.md](./models/hunyuan_fused_moe.py.md) | HunyuanFusedMoE |
| [worker/index.md](./worker/index.md) | Worker 索引 |
| [worker/npu_model_runner.py.md](./worker/npu_model_runner.py.md) | 基础 ModelRunner |
| [worker/npu_ar_model_runner.py.md](./worker/npu_ar_model_runner.py.md) | AR ModelRunner |
| [worker/npu_ar_worker.py.md](./worker/npu_ar_worker.py.md) | AR Worker |
| [worker/npu_generation_model_runner.py.md](./worker/npu_generation_model_runner.py.md) | Generation ModelRunner |
| [worker/npu_generation_worker.py.md](./worker/npu_generation_worker.py.md) | Generation Worker |
| [stage_configs/index.md](./stage_configs/index.md) | 阶段配置 |

## 架构特点

1. **独立 Worker 体系**：NPU 不复用通用 GPU Worker，而是有完整的 NPU 专用 Worker/ModelRunner 栈。
2. **继承 vllm-ascend**：依赖 `vllm_ascend` 包（`NPUPlatform`、`NPUWorker`、`NPUModelRunner`）。
3. **专用模型算子**：为 Ascend NPU 优化的 MoE 算子实现。
4. **丰富的阶段配置**：包含同步/异步块传输、TTS 独立配置等多种场景。
5. **不支持 torch.compile**：`supports_torch_inductor()` 返回 `False`。
