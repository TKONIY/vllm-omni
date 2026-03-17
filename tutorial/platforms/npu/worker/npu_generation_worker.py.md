# `npu_generation_worker.py` -- NPU 生成 Worker

## 文件概述

`npu_generation_worker.py` 定义了 `NPUGenerationWorker`，是 NPU 平台上 code2wav 等非自回归生成阶段的 Worker 入口。

## 关键代码解析

```python
class NPUGenerationWorker(OmniWorkerMixin, NPUWorker):
    """NPU generation worker for code2wav stage in Omni model."""

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        self.model_runner = NPUGenerationModelRunner(self.vllm_config, self.device)
```

结构与 `NPUARWorker` 完全对称，唯一差异是创建的 ModelRunner 类型为 `NPUGenerationModelRunner`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `NPUGenerationWorker` | 类 | NPU 生成 Worker |
| `init_device()` | 方法 | 初始化设备并创建 Generation ModelRunner |

## 与其他模块的关系

- **继承**：`OmniWorkerMixin` + `NPUWorker`
- **创建的 ModelRunner**：`NPUGenerationModelRunner`
- **注册位置**：`NPUOmniPlatform.get_omni_generation_worker_cls()`

## 总结

`NPUGenerationWorker` 与 `NPUARWorker` 结构对称，差异仅在于使用 `NPUGenerationModelRunner` 处理非自回归生成任务。
