# `__init__.py` — DreamID-Omni 模块初始化与导出

## 文件概述

该文件是 DreamID-Omni 视频+音频联合生成模型子包的入口文件，仅导出核心管线类 `DreamIDOmniPipeline`。

## 关键代码解析

```python
from .pipeline_dreamid_omni import DreamIDOmniPipeline

__all__ = ["DreamIDOmniPipeline"]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DreamIDOmniPipeline` | 类 | 视频+音频联合生成管线 |

## 与其他模块的关系

- `FusionModel` 和 `WanModel` 等内部组件不直接导出，通过管线间接使用
- 依赖外部包 `dreamid_omni` 提供基础工具

## 总结

`__init__.py` 仅导出 `DreamIDOmniPipeline`，保持简洁的公共 API。
