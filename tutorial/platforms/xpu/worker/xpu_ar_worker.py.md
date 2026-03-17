# `xpu_ar_worker.py` -- XPU 自回归 Worker

## 文件概述

`xpu_ar_worker.py` 定义了 `XPUARWorker`，是 XPU 平台上 thinker/talker 阶段的 Worker 入口。

## 关键代码解析

```python
class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        super().init_device()
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)
```

类继承关系：
- `OmniWorkerMixin`：Omni 多模态 Worker 通用功能
- `XPUWorker`：来自 vLLM，提供 XPU 设备初始化

`init_device()` 方法先调用 `XPUWorker.init_device()` 完成设备初始化，再创建 `XPUARModelRunner`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `XPUARWorker` | 类 | XPU 自回归 Worker |
| `init_device()` | 方法 | 初始化设备并创建 AR ModelRunner |

## 与其他模块的关系

- **继承**：`OmniWorkerMixin` + `vllm.v1.worker.xpu_worker.XPUWorker`
- **创建的 ModelRunner**：`XPUARModelRunner`
- **注册位置**：`XPUOmniPlatform.get_omni_ar_worker_cls()`

## 总结

`XPUARWorker` 是 XPU 平台自回归推理的入口，结构简洁，通过多重继承组合 Omni 和 XPU 能力。
