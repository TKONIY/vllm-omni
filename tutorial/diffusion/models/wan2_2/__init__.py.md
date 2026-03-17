# `__init__.py` — Wan2.2 模块初始化与导出

## 文件概述

该文件是 Wan2.2 视频/图像生成模型子包的入口文件，导出三种管线变体和核心 Transformer 模型。Wan2.2 是一个通用的文本到视频/图像生成框架，支持 T2V、I2V（图像到视频）和 TI2V（文本+图像到视频）。

## 关键代码解析

```python
from .pipeline_wan2_2 import (Wan22Pipeline, create_transformer_from_config, ...)
from .pipeline_wan2_2_i2v import (Wan22I2VPipeline, ...)
from .pipeline_wan2_2_ti2v import (Wan22TI2VPipeline, ...)
from .wan2_2_transformer import WanTransformer3DModel
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `WanTransformer3DModel` | 类 | 3D Transformer 核心模型 |
| `Wan22Pipeline` | 类 | 文本到视频/图像管线 |
| `Wan22I2VPipeline` | 类 | 图像到视频管线 |
| `Wan22TI2VPipeline` | 类 | 文本+图像到视频管线 |
| `load_transformer_config` | 函数 | 加载 Transformer 配置 |
| `create_transformer_from_config` | 函数 | 从配置创建 Transformer |
| `retrieve_latents` | 函数 | VAE 潜在变量提取 |

## 与其他模块的关系

- Wan2.2 是 Helios 和 DreamID-Omni 等模型的基础架构
- 三种管线共享同一个 `WanTransformer3DModel`

## 总结

`__init__.py` 将 Wan2.2 的三种管线变体和核心 Transformer 模型集中导出，形成完整的 T2V/I2V/TI2V 生成框架。
