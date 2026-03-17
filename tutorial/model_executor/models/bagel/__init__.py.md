# `__init__.py` — Bagel 模块入口

## 文件概述

这是 Bagel 模型目录的初始化文件，负责导出核心模型类 `OmniBagelForConditionalGeneration`，使其可以被 vLLM-omni 的模型注册系统识别和加载。

## 关键代码解析

```python
from .bagel import OmniBagelForConditionalGeneration

__all__ = ["OmniBagelForConditionalGeneration"]
```

仅导出一个类，保持接口简洁。

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `OmniBagelForConditionalGeneration` | 类 | Bagel 条件生成模型的 Omni 版本 |

## 与其他模块的关系

- 被 `vllm_omni` 的模型注册表 (`OmniModelRegistry`) 引用，用于根据模型架构名动态加载模型
- 实际实现位于 `bagel.py`

## 总结

简单的模块入口文件，职责是将内部实现类暴露为公共 API。
