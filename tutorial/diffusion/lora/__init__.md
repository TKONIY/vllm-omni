# `__init__.py` — LoRA 模块入口

## 文件概述

`lora/__init__.py` 是扩散模型 LoRA 子模块的入口文件。它仅导出一个核心类 `DiffusionLoRAManager`，作为整个 LoRA 适配器管理功能的统一对外接口。

## 关键代码解析

```python
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

__all__ = ["DiffusionLoRAManager"]
```

该文件的作用非常简洁：从 `manager` 模块中导入 `DiffusionLoRAManager`，并通过 `__all__` 声明其为唯一的公开 API。外部代码可通过以下方式使用：

```python
from vllm_omni.diffusion.lora import DiffusionLoRAManager
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionLoRAManager` | 类（重导出） | 扩散模型 LoRA 适配器管理器，负责加载、缓存、激活和切换 LoRA 适配器 |

## 与其他模块的关系

- **上游依赖**：直接依赖 `manager.py`，从中导入 `DiffusionLoRAManager`。
- **下游使用**：被扩散推理流水线（pipeline）调用，用于在推理过程中动态管理 LoRA 适配器。

## 总结

此文件是 LoRA 子模块的包入口，通过简洁的重导出模式，将 `DiffusionLoRAManager` 暴露为该子模块的唯一公开接口，使外部使用者无需关心内部模块结构。
