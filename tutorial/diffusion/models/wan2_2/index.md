# Wan2.2 模型模块索引

## 概述

Wan2.2 是一个通用的视频/图像生成框架，采用 3D DiT 架构，支持文本到视频（T2V）、图像到视频（I2V）和文本+图像到视频（TI2V）三种生成模式。它是 Helios 和 DreamID-Omni 等扩展模型的基础架构。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块初始化，导出三种管线和 Transformer |
| [`wan2_2_transformer.py`](wan2_2_transformer.py.md) | 3D Transformer 去噪模型（TP 并行 + 序列并行） |
| [`pipeline_wan2_2.py`](pipeline_wan2_2.py.md) | T2V/T2I 基础管线 |
| [`pipeline_wan2_2_i2v.py`](pipeline_wan2_2_i2v.py.md) | I2V 管线（+CLIP 图像编码） |
| [`pipeline_wan2_2_ti2v.py`](pipeline_wan2_2_ti2v.py.md) | TI2V 管线（双条件注入） |

## 架构关系

```
Wan2.2 管线系列
  ├── Wan22Pipeline (T2V/T2I)
  │     ├── UMT5EncoderModel (文本编码)
  │     ├── AutoencoderKLWan (VAE)
  │     └── WanTransformer3DModel
  │
  ├── Wan22I2VPipeline (I2V)
  │     ├── UMT5EncoderModel
  │     ├── CLIPVisionModelWithProjection (图像编码)
  │     ├── AutoencoderKLWan
  │     └── WanTransformer3DModel (+图像交叉注意力)
  │
  └── Wan22TI2VPipeline (TI2V)
        ├── UMT5EncoderModel
        ├── CLIPVisionModelWithProjection
        ├── AutoencoderKLWan (图像编码 + 首帧条件)
        └── WanTransformer3DModel (+图像交叉注意力)

WanTransformer3DModel
  ├── WanRotaryPosEmbed (3D RoPE)
  ├── WanTimeTextImageEmbedding (条件嵌入)
  ├── Conv3dLayer (Patch embedding)
  └── WanTransformerBlock x N
        ├── WanSelfAttention (QKV 融合 + TP)
        ├── WanCrossAttention (文本 + 可选图像)
        └── WanFeedForward (TP)
```
