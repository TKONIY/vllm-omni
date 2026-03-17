# `__init__.py` — hooks 包入口

## 文件概述

`hooks/__init__.py` 汇总导出了 hooks 子包的核心接口，包括基础 hook 机制（`BaseState`、`StateManager`、`ModelHook`、`HookRegistry`）和序列并行 hooks（`SequenceParallelSplitHook`、`SequenceParallelGatherHook` 及相关函数）。

## 关键代码解析

```python
from vllm_omni.diffusion.hooks.base import BaseState, HookRegistry, ModelHook, StateManager
from vllm_omni.diffusion.hooks.sequence_parallel import (
    SequenceParallelGatherHook,
    SequenceParallelSplitHook,
    apply_sequence_parallel,
    disable_sequence_parallel_for_model,
    enable_sequence_parallel_for_model,
    remove_sequence_parallel,
)
```

## 核心类/函数

| 名称 | 来源 | 说明 |
|------|------|------|
| `ModelHook` | `base.py` | Hook 基类 |
| `HookRegistry` | `base.py` | Hook 注册管理器 |
| `SequenceParallelSplitHook` | `sequence_parallel.py` | 序列并行输入分片 Hook |
| `SequenceParallelGatherHook` | `sequence_parallel.py` | 序列并行输出聚合 Hook |
| `apply_sequence_parallel` | `sequence_parallel.py` | 应用序列并行 hooks |

## 总结

该文件作为 hooks 子包的公共 API 入口，将基础 hook 机制和序列并行 hooks 统一暴露，方便外部模块按需导入。
