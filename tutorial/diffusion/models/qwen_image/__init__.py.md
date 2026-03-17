# `__init__.py` — QwenImage 模块初始化与导出

## 文件概述

该文件是 QwenImage 图像生成模型子包的入口文件，导出核心组件。QwenImage 系列包括文本到图像生成、图像编辑、增强编辑和分层生成四种管线变体。

## 关键代码解析

```python
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import QwenImageCFGParallelMixin
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline, get_qwen_image_post_process_func,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `QwenImageCFGParallelMixin` | 类 | CFG 并行 Mixin（所有管线共享） |
| `QwenImagePipeline` | 类 | 文本到图像生成管线 |
| `QwenImageTransformer2DModel` | 类 | 2D Transformer 模型 |
| `get_qwen_image_post_process_func` | 函数 | 图像后处理函数工厂 |

## 与其他模块的关系

- 其他管线（Edit/EditPlus/Layered）需通过各自模块直接导入
- `QwenImageCFGParallelMixin` 被所有管线变体继承

## 总结

`__init__.py` 导出了 QwenImage 系列的核心基础组件。各编辑管线因参数和依赖不同，需从各自模块单独导入。
