# `__init__.py` — 模型加载器入口

## 文件概述

`model_loader/__init__.py` 是模型加载器子模块的入口文件。当前该文件为空（仅包含空行），模块的核心功能由 `diffusers_loader.py` 提供。

## 关键代码解析

该文件没有实际代码内容。模块的使用方式是直接导入具体的加载器类：

```python
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
```

## 核心类/函数

无。

## 与其他模块的关系

- **`diffusers_loader.py`**：模块的核心实现。
- **`gguf_adapters/`**：GGUF 格式适配层。

## 总结

此入口文件当前为空，模块功能通过直接导入子文件来使用。
