# `mixins.py` — Omni Worker 插件加载混入

## 文件概述

`mixins.py` 定义了 `OmniWorkerMixin`，一个极为轻量的 Mixin 类。它的唯一职责是确保 Worker 进程在初始化时加载 vLLM-Omni 的通用插件。

## 关键代码解析

```python
class OmniWorkerMixin:
    """Mixin to ensure Omni plugins are loaded in worker processes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
```

### 工作机制

1. `OmniWorkerMixin` 通过 Python 的多重继承机制（MRO）工作。在 `GPUARWorker` 和 `GPUGenerationWorker` 中，它被放在继承列表的前面：

```python
class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    ...
```

2. 当 Worker 实例化时，`OmniWorkerMixin.__init__` 先于 `OmniGPUWorkerBase.__init__` 被调用（按 MRO 顺序），通过 `super().__init__()` 链式调用后续基类的构造函数。

3. `load_omni_general_plugins()` 在 Worker 进程中注册 Omni 特有的插件和扩展，例如自定义的模型注册、自定义 attention 后端等。

### 为什么需要这个 Mixin

在分布式推理中，Worker 通常运行在独立的进程（通过 Ray 或其他 executor 启动）。这些进程不会自动继承主进程中已加载的插件状态，因此需要在每个 Worker 进程的初始化阶段显式加载插件。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniWorkerMixin` | 类 (Mixin) | 在 Worker 初始化时加载 Omni 通用插件 |

## 与其他模块的关系

- **被混入**：`GPUARWorker` 和 `GPUGenerationWorker` 都将其作为第一个基类混入
- **依赖**：`vllm_omni.plugins.load_omni_general_plugins`，执行实际的插件加载逻辑

## 总结

`OmniWorkerMixin` 虽然只有几行代码，但在分布式架构中扮演着不可或缺的角色。它保证了每个 Worker 进程都正确加载了 Omni 插件，使得自定义模型、自定义 attention 等扩展在所有 Worker 中可用。
