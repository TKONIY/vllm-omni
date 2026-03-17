# `npu_ar_worker.py` -- NPU 自回归 Worker

## 文件概述

`npu_ar_worker.py` 定义了 `NPUARWorker`，是 NPU 平台上 thinker/talker 阶段的 Worker 入口。它负责设备初始化和 `NPUARModelRunner` 的创建。

## 关键代码解析

```python
class NPUARWorker(OmniWorkerMixin, NPUWorker):
    """NPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        self.model_runner = NPUARModelRunner(self.vllm_config, self.device)
```

类继承关系：
- `OmniWorkerMixin`：提供 Omni 多模态 Worker 的通用功能（附加信息处理等）
- `NPUWorker`：来自 `vllm_ascend`，提供 NPU 设备的初始化逻辑

`init_device()` 方法：
1. `self._init_device()`：初始化 NPU 设备（设置 HCCL 通信等）
2. `init_workspace_manager`：初始化工作空间管理器
3. 创建 `NPUARModelRunner` 实例

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `NPUARWorker` | 类 | NPU 自回归 Worker |
| `init_device()` | 方法 | 初始化设备并创建 ModelRunner |

## 与其他模块的关系

- **继承**：`OmniWorkerMixin`（Omni 通用逻辑）+ `NPUWorker`（vllm_ascend 设备管理）
- **创建的 ModelRunner**：`NPUARModelRunner`
- **注册位置**：通过 `NPUOmniPlatform.get_omni_ar_worker_cls()` 返回该类的全限定名

## 总结

`NPUARWorker` 是一个简洁的组合类，通过多重继承将 Omni 多模态 Worker 能力和 NPU 设备管理能力结合在一起。其核心职责是设备初始化和 ModelRunner 创建。
