# `parallel/__init__.py` — 并行注意力策略包初始化

## 文件概述

`parallel/__init__.py` 是并行注意力策略包的初始化文件，导出了核心接口和工厂函数。该包提供与注意力内核后端正交的通信/重分片策略。

## 关键代码解析

```python
"""Parallel attention strategies.

This package provides **communication / resharding strategies** for attention,
orthogonal to the **attention kernel backend** (SDPA/Flash/Sage).

The goal is to keep `vllm_omni.diffusion.attention.layer.Attention` small and
extensible: adding a new parallelism method should not require editing the core
Attention module, only adding a new strategy and selecting it in the factory.
"""

from .base import NoParallelAttention, ParallelAttentionContext, ParallelAttentionStrategy
from .factory import build_parallel_attention_strategy

__all__ = [
    "ParallelAttentionStrategy",
    "ParallelAttentionContext",
    "NoParallelAttention",
    "build_parallel_attention_strategy",
]
```

设计原则：
- **与内核后端正交**：并行策略决定 Q/K/V 和输出如何在设备间分片/通信，而内核后端决定如何计算注意力
- **可扩展性**：新增并行方式只需添加新策略类并在工厂中注册，无需修改 `Attention` 类

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ParallelAttentionStrategy` | Protocol | 并行策略的接口协议 |
| `ParallelAttentionContext` | 数据类 | 前向过程中策略传递的上下文 |
| `NoParallelAttention` | 类 | 默认策略：不做任何并行通信 |
| `build_parallel_attention_strategy` | 函数 | 根据配置构建并行策略的工厂函数 |

## 与其他模块的关系

- **`base.py`**：定义接口和默认实现
- **`factory.py`**：工厂函数实现
- **`ring.py`**：Ring 并行策略
- **`ulysses.py`**：Ulysses 并行策略
- **`layer.py`**：`Attention` 类使用此包构建和执行并行策略

## 总结

该包通过策略模式将并行通信逻辑从 `Attention` 类中解耦，导出了核心接口和工厂函数，为扩展新的并行方式提供了清晰的入口。
