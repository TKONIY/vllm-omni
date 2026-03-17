# `xpu_generation_worker.py` -- XPU 生成 Worker

## 文件概述

`xpu_generation_worker.py` 定义了 `XPUGenerationWorker`，是 XPU 平台上 code2wav 等非自回归生成阶段的 Worker 入口。

## 关键代码解析

```python
class XPUGenerationWorker(OmniWorkerMixin, XPUWorker):
    """XPU generation worker for the code2wav (non-AR waveform generation) stage in the Omni model."""

    def init_device(self):
        super().init_device()
        self.model_runner: XPUGenerationModelRunner = XPUGenerationModelRunner(self.vllm_config, self.device)
```

结构与 `XPUARWorker` 完全对称，创建 `XPUGenerationModelRunner`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `XPUGenerationWorker` | 类 | XPU 生成 Worker |
| `init_device()` | 方法 | 初始化设备并创建 Generation ModelRunner |

## 与其他模块的关系

- **继承**：`OmniWorkerMixin` + `XPUWorker`
- **创建的 ModelRunner**：`XPUGenerationModelRunner`
- **注册位置**：`XPUOmniPlatform.get_omni_generation_worker_cls()`

## 总结

`XPUGenerationWorker` 是 XPU 平台非自回归生成推理的入口，与 AR Worker 结构对称。
