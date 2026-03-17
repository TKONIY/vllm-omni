# `utils.py` -- 分布式工具函数

## 文件概述

`utils.py` 提供了一个简单的工具函数 `get_local_device`，用于根据当前 rank 获取对应的 torch 设备。它是分布式环境中设备管理的基础工具。

## 关键代码解析

```python
import os
import torch
from vllm_omni.platforms import current_omni_platform

def get_local_device() -> torch.device:
    """根据检测到的设备类型返回当前 rank 的 torch 设备。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return current_omni_platform.get_torch_device(local_rank)
```

该函数通过 `LOCAL_RANK` 环境变量确定本地 rank，然后使用平台抽象层 `current_omni_platform` 获取对应的设备。这使得代码在 CUDA 和 NPU 等不同硬件平台上都能正常工作。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_local_device()` | 函数 | 返回当前 rank 对应的 torch.device |

## 与其他模块的关系

- **platforms 模块**: 使用 `current_omni_platform` 进行设备抽象
- **其他分布式模块**: 在需要获取本地设备时调用此函数

## 总结

该文件是一个轻量级工具模块，通过平台抽象层实现了跨硬件平台的设备获取。`LOCAL_RANK` 环境变量是多 GPU 训练/推理框架的标准约定，默认值 0 确保了单 GPU 场景的兼容性。
