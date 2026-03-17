# `__init__.py` — 模块初始化

## 文件概述

导出 Qwen3-Omni MoE 的统一入口类。

## 关键代码解析

```python
from .qwen3_omni import Qwen3OmniMoeForConditionalGeneration
__all__ = ["Qwen3OmniMoeForConditionalGeneration"]
```

仅导出一个类，即多阶段统一入口模型。

## 与其他模块的关系

使得外部可以通过 `from vllm_omni.model_executor.models.qwen3_omni import Qwen3OmniMoeForConditionalGeneration` 直接引用。

## 总结

标准的包初始化文件，导出统一入口类供 vLLM 模型注册表使用。
