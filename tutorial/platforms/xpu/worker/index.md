# xpu/worker/ 子模块索引

## 模块概述

`xpu/worker/` 包含 XPU 平台专用的 Worker 和 ModelRunner 实现。与 NPU 平台的深度定制不同，XPU Worker 采用**轻量级包装**策略，直接继承通用 GPU 实现，仅覆盖设备初始化和同步方法。

## 类继承关系

```
GPUARModelRunner (vllm_omni.worker)
        |
XPUARModelRunner     <-- 轻量级包装，仅覆盖设备方法
        |
XPUARWorker          <-- 使用 XPUARModelRunner

GPUGenerationModelRunner (vllm_omni.worker)
        |
XPUGenerationModelRunner  <-- 轻量级包装
        |
XPUGenerationWorker       <-- 使用 XPUGenerationModelRunner
```

## 文件列表

| 文件 | 说明 |
|------|------|
| `__init__.py` | 空初始化文件 |
| [xpu_ar_model_runner.py.md](./xpu_ar_model_runner.py.md) | XPU AR ModelRunner |
| [xpu_ar_worker.py.md](./xpu_ar_worker.py.md) | XPU AR Worker |
| [xpu_generation_model_runner.py.md](./xpu_generation_model_runner.py.md) | XPU Generation ModelRunner |
| [xpu_generation_worker.py.md](./xpu_generation_worker.py.md) | XPU Generation Worker |

## 设计特点

XPU Worker 的核心设计是通过 `torch_cuda_wrapper` 上下文管理器使 GPU 通用代码透明运行在 XPU 上。每个 ModelRunner 仅需覆盖三个方法：
1. `__init__`：在 `torch_cuda_wrapper` 上下文中调用父类初始化
2. `_init_device_properties`：XPU 不需要 SM 数量信息
3. `_sync_device`：使用 `torch.xpu.synchronize()`
