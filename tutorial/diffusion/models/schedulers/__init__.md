# `schedulers/__init__.py` -- 调度器包初始化

## 文件概述

调度器子包的入口文件，导出 `FlowUniPCMultistepScheduler` 类供外部使用。

**文件路径**: `vllm_omni/diffusion/models/schedulers/__init__.py`

## 关键代码解析

```python
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)

__all__ = [
    "FlowUniPCMultistepScheduler",
]
```

## 核心类/函数

| 名称 | 来源 | 说明 |
|------|------|------|
| `FlowUniPCMultistepScheduler` | `scheduling_flow_unipc_multistep.py` | 基于 Flow Matching 的 UniPC 多步调度器 |

## 与其他模块的关系

本包被需要自定义调度器的 Pipeline（如 Wan2.2 视频生成模型）使用。

## 总结

调度器包以简洁的方式导出核心调度器类，为扩散模型提供高效的采样策略。
