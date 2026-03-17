# `__init__.py` — 入口模块初始化

## 文件概述

该文件是 `entrypoints` 包的入口，负责导出三个核心入口类，方便外部使用者直接从 `vllm_omni.entrypoints` 导入。

## 关键代码解析

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.omni import Omni

__all__ = [
    "AsyncOmni",
    "AsyncOmniDiffusion",
    "Omni",
]
```

导出的三个类分别对应三种使用场景：
- `AsyncOmni`: 异步在线服务（多阶段 LLM 管线）
- `AsyncOmniDiffusion`: 异步扩散模型推理
- `Omni`: 同步离线批量推理

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `AsyncOmni` | 类 (re-export) | 异步多阶段推理入口 |
| `AsyncOmniDiffusion` | 类 (re-export) | 异步扩散模型入口 |
| `Omni` | 类 (re-export) | 同步批量推理入口 |

## 与其他模块的关系

该文件仅做 re-export，实际实现分别位于 `async_omni.py`、`async_omni_diffusion.py` 和 `omni.py`。

## 总结

一个标准的包初始化文件，将三个核心入口类统一导出，简化外部导入路径。
