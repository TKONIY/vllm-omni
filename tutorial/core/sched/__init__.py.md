# `core/sched/__init__.py` — 调度器模块入口

## 文件概述

调度器子模块的入口文件，导出三个核心类。

## 关键代码解析

```python
from .omni_ar_scheduler import OmniARScheduler
from .omni_generation_scheduler import OmniGenerationScheduler
from .output import OmniNewRequestData
```

## 导出列表

| 名称 | 用途 |
|------|------|
| `OmniARScheduler` | 自回归模型调度器 |
| `OmniGenerationScheduler` | 生成/扩散模型调度器 |
| `OmniNewRequestData` | 新请求调度数据 |

## 总结

标准的模块入口文件，提供统一的调度器导入路径。
