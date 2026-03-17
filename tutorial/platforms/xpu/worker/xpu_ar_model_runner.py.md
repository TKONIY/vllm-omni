# `xpu_ar_model_runner.py` -- XPU 自回归 ModelRunner

## 文件概述

`xpu_ar_model_runner.py` 实现了 `XPUARModelRunner`，是 XPU 平台上自回归推理的 ModelRunner。采用轻量级包装策略，直接继承通用 `GPUARModelRunner`，仅覆盖设备相关方法。

## 关键代码解析

```python
class XPUARModelRunner(GPUARModelRunner):
    def __init__(self, *args, **kwargs):
        with torch_cuda_wrapper():
            super().__init__(*args, **kwargs)

    def _init_device_properties(self):
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()
```

### 1. 初始化

在 `torch_cuda_wrapper()` 上下文中调用父类 `GPUARModelRunner.__init__`，使初始化过程中的 CUDA Stream API 调用被自动重定向到 XPU。

### 2. 设备属性

XPU 没有 SM（Streaming Multiprocessor）的概念，因此设置 `num_sms = None`。

### 3. 设备同步

使用 `torch.xpu.synchronize()` 替代 `torch.cuda.synchronize()`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `XPUARModelRunner` | 类 | XPU 自回归 ModelRunner |
| `_init_device_properties()` | 方法 | 设置设备属性（num_sms=None） |
| `_sync_device()` | 方法 | XPU 设备同步 |

## 与其他模块的关系

- **基类**：`GPUARModelRunner`（通用 GPU 实现）
- **依赖**：`torch_cuda_wrapper`（xpu/utils.py）
- **使用者**：`XPUARWorker`

## 总结

`XPUARModelRunner` 展示了 XPU 平台的轻量级适配策略。通过仅覆盖三个方法并使用 CUDA API 兼容层，它以极少的代码量实现了完整的自回归推理能力。
