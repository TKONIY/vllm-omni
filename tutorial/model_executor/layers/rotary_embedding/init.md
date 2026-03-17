# `rotary_embedding/__init__.py` -- 旋转位置编码模块导出

## 文件概述

该文件是 `rotary_embedding` 子模块的入口，负责导出 `OmniMRotaryEmbedding` 类，使其可以通过简洁的导入路径使用。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/layers/rotary_embedding/__init__.py`

## 关键代码解析

```python
from .mrope import OmniMRotaryEmbedding

__all__ = ["OmniMRotaryEmbedding"]
```

文件结构非常简单：从 `mrope.py` 导入 `OmniMRotaryEmbedding` 类，并通过 `__all__` 限定公开导出的符号。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniMRotaryEmbedding` | 类（导出） | 多模态旋转位置编码扩展类 |

## 与其他模块的关系

- **mrope.py**: 实际实现所在
- **models/**: 通过此入口导入 `OmniMRotaryEmbedding`

## 总结

标准的 Python 包初始化文件，提供简洁的导入路径 `from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding`。
