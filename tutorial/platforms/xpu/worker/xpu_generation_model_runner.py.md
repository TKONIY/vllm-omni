# `xpu_generation_model_runner.py` -- XPU 生成 ModelRunner

## 文件概述

`xpu_generation_model_runner.py` 实现了 `XPUGenerationModelRunner`，用于 XPU 平台上的非自回归生成阶段（如 code2wav）。与 `XPUARModelRunner` 一样采用轻量级包装策略。

## 关键代码解析

```python
class XPUGenerationModelRunner(GPUGenerationModelRunner):
    def __init__(self, *args, **kwargs):
        with torch_cuda_wrapper():
            super().__init__(*args, **kwargs)

    def _init_device_properties(self):
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()
```

结构与 `XPUARModelRunner` 完全对称，唯一差异是基类为 `GPUGenerationModelRunner`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `XPUGenerationModelRunner` | 类 | XPU 非自回归生成 ModelRunner |
| `_init_device_properties()` | 方法 | 设置 num_sms=None |
| `_sync_device()` | 方法 | XPU 设备同步 |

## 与其他模块的关系

- **基类**：`GPUGenerationModelRunner`（通用 GPU 生成实现）
- **依赖**：`torch_cuda_wrapper`（xpu/utils.py）
- **使用者**：`XPUGenerationWorker`

## 总结

`XPUGenerationModelRunner` 与 AR 版本结构对称，以极少代码量将 GPU 生成 ModelRunner 适配到 XPU 设备。
