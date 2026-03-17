# platforms/ 模块教程索引

## 模块概述

`platforms/` 模块是 vllm-omni 项目的**硬件平台抽象层**，负责将多模态推理引擎适配到不同的硬件加速器上运行。该模块采用插件化架构，通过统一的 `OmniPlatform` 接口屏蔽底层硬件差异，使上层业务逻辑无需关心具体运行在哪种设备上。

## 目录结构

```
platforms/
├── __init__.py              # 平台自动检测与懒加载入口
├── interface.py             # OmniPlatform 抽象基类定义
├── cuda/                    # NVIDIA CUDA/GPU 平台实现
│   ├── __init__.py
│   └── platform.py
├── rocm/                    # AMD ROCm/GPU 平台实现
│   ├── __init__.py
│   ├── platform.py
│   └── stage_configs/       # ROCm 专用阶段配置
├── npu/                     # 华为 Ascend NPU 平台实现
│   ├── __init__.py
│   ├── platform.py
│   ├── models/              # NPU 专用模型算子
│   ├── worker/              # NPU 专用 Worker 与 ModelRunner
│   └── stage_configs/       # NPU 专用阶段配置
└── xpu/                     # Intel XPU 平台实现
    ├── __init__.py
    ├── platform.py
    ├── utils.py
    ├── worker/              # XPU 专用 Worker 与 ModelRunner
    └── stage_configs/       # XPU 专用阶段配置
```

## 架构设计

平台模块采用**双重继承**设计：

1. **OmniPlatform（接口层）**：继承自 vLLM 的 `Platform`，新增 Omni 多模态推理专用接口（如 `get_omni_ar_worker_cls`、`get_diffusion_attn_backend_cls` 等）。
2. **具体平台类**：同时继承 `OmniPlatform` 和对应 vLLM 平台基类（如 `CudaPlatformBase`、`RocmPlatform`、`NPUPlatform`、`XPUPlatform`），获得硬件操作能力的同时实现 Omni 接口。

## 教程文件列表

### 顶层文件
| 文件 | 说明 |
|------|------|
| [__init__.py.md](./\_\_init\_\_.py.md) | 平台自动检测与懒加载机制 |
| [interface.py.md](./interface.py.md) | OmniPlatform 抽象基类 |

### CUDA 平台
| 文件 | 说明 |
|------|------|
| [cuda/index.md](./cuda/index.md) | CUDA 子模块索引 |
| [cuda/__init__.py.md](./cuda/__init__.py.md) | CUDA 包导出 |
| [cuda/platform.py.md](./cuda/platform.py.md) | CudaOmniPlatform 实现 |

### ROCm 平台
| 文件 | 说明 |
|------|------|
| [rocm/index.md](./rocm/index.md) | ROCm 子模块索引 |
| [rocm/__init__.py.md](./rocm/__init__.py.md) | ROCm 包导出 |
| [rocm/platform.py.md](./rocm/platform.py.md) | RocmOmniPlatform 实现 |
| [rocm/stage_configs/index.md](./rocm/stage_configs/index.md) | ROCm 阶段配置说明 |

### NPU 平台
| 文件 | 说明 |
|------|------|
| [npu/index.md](./npu/index.md) | NPU 子模块索引 |
| [npu/__init__.py.md](./npu/__init__.py.md) | NPU 包导出 |
| [npu/platform.py.md](./npu/platform.py.md) | NPUOmniPlatform 实现 |
| [npu/models/index.md](./npu/models/index.md) | NPU 专用模型算子索引 |
| [npu/models/hunyuan_fused_moe.py.md](./npu/models/hunyuan_fused_moe.py.md) | HunyuanFusedMoE NPU 实现 |
| [npu/worker/index.md](./npu/worker/index.md) | NPU Worker 索引 |
| [npu/worker/npu_model_runner.py.md](./npu/worker/npu_model_runner.py.md) | NPU 基础 ModelRunner |
| [npu/worker/npu_ar_model_runner.py.md](./npu/worker/npu_ar_model_runner.py.md) | NPU 自回归 ModelRunner |
| [npu/worker/npu_ar_worker.py.md](./npu/worker/npu_ar_worker.py.md) | NPU 自回归 Worker |
| [npu/worker/npu_generation_model_runner.py.md](./npu/worker/npu_generation_model_runner.py.md) | NPU 生成 ModelRunner |
| [npu/worker/npu_generation_worker.py.md](./npu/worker/npu_generation_worker.py.md) | NPU 生成 Worker |
| [npu/stage_configs/index.md](./npu/stage_configs/index.md) | NPU 阶段配置说明 |

### XPU 平台
| 文件 | 说明 |
|------|------|
| [xpu/index.md](./xpu/index.md) | XPU 子模块索引 |
| [xpu/__init__.py.md](./xpu/__init__.py.md) | XPU 包导出 |
| [xpu/platform.py.md](./xpu/platform.py.md) | XPUOmniPlatform 实现 |
| [xpu/utils.py.md](./xpu/utils.py.md) | XPU 工具函数 |
| [xpu/worker/index.md](./xpu/worker/index.md) | XPU Worker 索引 |
| [xpu/worker/xpu_ar_model_runner.py.md](./xpu/worker/xpu_ar_model_runner.py.md) | XPU 自回归 ModelRunner |
| [xpu/worker/xpu_ar_worker.py.md](./xpu/worker/xpu_ar_worker.py.md) | XPU 自回归 Worker |
| [xpu/worker/xpu_generation_model_runner.py.md](./xpu/worker/xpu_generation_model_runner.py.md) | XPU 生成 ModelRunner |
| [xpu/worker/xpu_generation_worker.py.md](./xpu/worker/xpu_generation_worker.py.md) | XPU 生成 Worker |
| [xpu/stage_configs/index.md](./xpu/stage_configs/index.md) | XPU 阶段配置说明 |
