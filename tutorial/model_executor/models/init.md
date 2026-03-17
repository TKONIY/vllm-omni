# `models/__init__.py` -- 模型模块初始化与导出

## 文件概述

该文件是 `models/` 模块的入口，负责导出核心模型类和模型注册表实例。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/models/__init__.py`

## 关键代码解析

```python
from .bagel.bagel import OmniBagelForConditionalGeneration
from .qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from .registry import OmniModelRegistry  # noqa: F401

__all__ = [
    "Qwen3OmniMoeForConditionalGeneration",
    "OmniBagelForConditionalGeneration",
]
```

该文件完成三个任务：

1. **导入具体模型类**: `OmniBagelForConditionalGeneration` 和 `Qwen3OmniMoeForConditionalGeneration`
2. **导入注册表**: `OmniModelRegistry` 通过 `# noqa: F401` 标注为"虽然未在本文件使用但需要导入以触发注册"
3. **限定公开接口**: `__all__` 只暴露两个模型类

注意：虽然 `__all__` 只包含两个模型，但 `OmniModelRegistry` 包含了所有已注册模型（通过 `registry.py` 中的 `_OMNI_MODELS` 字典）。其他模型通过注册表按需懒加载。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3OmniMoeForConditionalGeneration` | 类（导出） | Qwen3-Omni MoE 条件生成模型 |
| `OmniBagelForConditionalGeneration` | 类（导出） | Bagel 条件生成模型 |
| `OmniModelRegistry` | 实例（导出） | 全局模型注册表 |

## 与其他模块的关系

- **registry.py**: `OmniModelRegistry` 的定义来源
- **bagel/bagel.py**: `OmniBagelForConditionalGeneration` 的实现
- **qwen3_omni/**: `Qwen3OmniMoeForConditionalGeneration` 的实现

## 总结

`models/__init__.py` 是模型模块的简洁入口，直接导出两个常用模型类并注册全局模型注册表。大多数模型通过 `OmniModelRegistry` 按架构名懒加载，无需在此显式导入。
