# `pipeline_mammothmoda2_dit.py` — DiT 管线兼容性 shim

## 文件概述

一个简单的兼容性桥接模块，将 `MammothModa2DiTPipeline` 从 `vllm_omni.diffusion` 包重新导出到当前路径，使 OmniModelRegistry 和下游代码能通过 `model_executor.models.mammoth_moda2` 路径访问 DiT 实现。

## 关键代码解析

```python
"""
Compatibility shim.
The MammothModa2 DiT implementation lives under `vllm_omni.diffusion` to align
with other ARDiT structured models.
"""
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
)

__all__ = ["MammothModa2DiTPipeline"]
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `MammothModa2DiTPipeline` | 类 | DiT 扩散管线（重导出） |

## 总结

纯重导出模块，实际 DiT 实现位于 `vllm_omni/diffusion/models/mammoth_moda2/` 目录下。
