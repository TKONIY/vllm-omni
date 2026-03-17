# xpu/ 子模块索引

## 模块概述

`xpu/` 子模块实现了基于 Intel XPU（如 Intel Arc GPU）的 OmniPlatform。XPU 平台的实现策略介于 CUDA 和 NPU 之间：它有自己的 Worker 和 ModelRunner，但采用轻量级包装方式，通过 CUDA API 兼容层将 GPU 通用实现适配到 XPU 设备上。

## 目录结构

```
xpu/
├── __init__.py           # 包导出
├── platform.py           # XPUOmniPlatform 实现
├── utils.py              # CUDA API 兼容层
├── worker/               # XPU Worker 和 ModelRunner
│   ├── __init__.py
│   ├── xpu_ar_model_runner.py
│   ├── xpu_ar_worker.py
│   ├── xpu_generation_model_runner.py
│   └── xpu_generation_worker.py
└── stage_configs/        # XPU 阶段配置
    ├── qwen2_5_omni.yaml
    └── qwen3_omni_moe.yaml
```

## 教程文件列表

| 文件 | 说明 |
|------|------|
| [__init__.py.md](./__init__.py.md) | 包导出 |
| [platform.py.md](./platform.py.md) | XPUOmniPlatform 实现 |
| [utils.py.md](./utils.py.md) | CUDA API 兼容层 |
| [worker/index.md](./worker/index.md) | Worker 索引 |
| [worker/xpu_ar_model_runner.py.md](./worker/xpu_ar_model_runner.py.md) | AR ModelRunner |
| [worker/xpu_ar_worker.py.md](./worker/xpu_ar_worker.py.md) | AR Worker |
| [worker/xpu_generation_model_runner.py.md](./worker/xpu_generation_model_runner.py.md) | Generation ModelRunner |
| [worker/xpu_generation_worker.py.md](./worker/xpu_generation_worker.py.md) | Generation Worker |
| [stage_configs/index.md](./stage_configs/index.md) | 阶段配置 |

## 架构特点

1. **轻量级适配**：XPU ModelRunner 通过 `torch_cuda_wrapper` 上下文管理器将 CUDA Stream API 替换为 XPU 等效 API。
2. **基于 GPU 通用实现**：直接继承 `GPUARModelRunner` / `GPUGenerationModelRunner`，仅覆盖设备相关方法。
3. **不支持 torch.compile**：`supports_torch_inductor()` 返回 `False`。
4. **分布式后端灵活**：支持 xccl 和 ccl 两种通信后端。
