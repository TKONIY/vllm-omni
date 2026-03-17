# `__init__.py` — 包初始化与导出

## 文件概述

该文件从子模块导入核心类和函数，并通过 `__all__` 定义公开 API。

## 关键代码解析

```python
from .stats import OrchestratorAggregator, StageRequestStats, StageStats
from .utils import count_tokens_from_outputs

__all__ = [
    "OrchestratorAggregator",
    "StageStats",
    "StageRequestStats",
    "count_tokens_from_outputs",
]
```

## 导出列表

| 名称 | 来源 | 说明 |
|------|------|------|
| `OrchestratorAggregator` | `stats.py` | 编排器指标聚合器 |
| `StageStats` | `stats.py` | 阶段总体统计 |
| `StageRequestStats` | `stats.py` | 单次请求的阶段统计 |
| `count_tokens_from_outputs` | `utils.py` | 从引擎输出统计 token 数 |

## 总结

标准的包初始化文件，集中导出本模块的核心 API，方便外部通过 `from vllm_omni.metrics import ...` 使用。
