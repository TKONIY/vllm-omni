# QwenImage 模型模块索引

## 概述

QwenImage 是一个基于 DiT（Diffusion Transformer）架构的图像生成系列，包括文本到图像生成、图像编辑、增强编辑和分层生成四种管线。使用 Qwen2.5-VL 作为文本编码器，FlowMatch Euler 调度器进行去噪。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块初始化，导出核心组件 |
| [`autoencoder_kl_qwenimage.py`](autoencoder_kl_qwenimage.py.md) | 因果 3D 卷积 KL 自编码器 |
| [`cfg_parallel.py`](cfg_parallel.py.md) | CFG 并行 Mixin（四种管线共享） |
| [`pipeline_qwen_image.py`](pipeline_qwen_image.py.md) | 文本到图像生成管线 |
| [`pipeline_qwen_image_edit.py`](pipeline_qwen_image_edit.py.md) | 图像编辑管线 |
| [`pipeline_qwen_image_edit_plus.py`](pipeline_qwen_image_edit_plus.py.md) | 增强图像编辑管线 |
| [`pipeline_qwen_image_layered.py`](pipeline_qwen_image_layered.py.md) | 分层图像生成管线 |
| [`qwen_image_transformer.py`](qwen_image_transformer.py.md) | 2D Transformer 去噪模型 |

## 架构关系

```
QwenImage 管线系列（共享 QwenImageCFGParallelMixin）
  ├── QwenImagePipeline (T2I)
  ├── QwenImageEditPipeline (编辑, +SupportImageInput)
  ├── QwenImageEditPlusPipeline (增强编辑, +SupportImageInput)
  └── QwenImageLayeredPipeline (分层生成, +SupportImageInput)
       │
       ├── Qwen2.5-VL (文本编码器)
       ├── DistributedAutoencoderKLQwenImage (VAE)
       │     ├── QwenImageEncoder3d (因果卷积编码器)
       │     └── QwenImageDecoder3d (解码器)
       ├── QwenImageTransformer2DModel (DiT)
       │     ├── QwenEmbedLayer3DRope (3D RoPE 嵌入)
       │     └── QwenImageTransformerBlock x N
       │           ├── 自注意力 + RoPE
       │           ├── 交叉注意力 (文本条件)
       │           └── FeedForward
       └── FlowMatchEulerDiscreteScheduler
```

## 管线变体对比

| 管线 | 输入 | 特点 |
|------|------|------|
| `QwenImagePipeline` | 文本 | 基础 T2I |
| `QwenImageEditPipeline` | 文本 + 图像 | 条件图像拼接 |
| `QwenImageEditPlusPipeline` | 文本 + 图像 | 增强预处理 |
| `QwenImageLayeredPipeline` | 文本 + 多层图像 | 图层合成 |
