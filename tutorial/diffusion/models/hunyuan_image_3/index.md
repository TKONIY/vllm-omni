# HunyuanImage3 模型模块索引

## 概述

HunyuanImage3 是一个基于自回归（AR）架构的统一多模态图像生成模型。它将文本和图像 token 统一在同一个 Decoder-only Transformer 中，结合 MoE 专家层和扩散去噪完成高质量图像生成，支持文本到图像和条件图像编辑。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块初始化，导出核心组件 |
| [`autoencoder.py`](autoencoder.py.md) | 3D 卷积 KL 自编码器（DCAE 架构，支持分块编解码） |
| [`hunyuan_fused_moe.py`](hunyuan_fused_moe.py.md) | 平台自适应融合 MoE 层工厂 |
| [`hunyuan_image_3_tokenizer.py`](hunyuan_image_3_tokenizer.py.md) | 多模态 Tokenizer（文本+图像+控制信号统一编排） |
| [`hunyuan_image_3_transformer.py`](hunyuan_image_3_transformer.py.md) | Transformer 模型、配置、注意力层、MoE 块、扩散管线 |
| [`pipeline_hunyuan_image_3.py`](pipeline_hunyuan_image_3.py.md) | 完整推理管线（整合所有组件的端到端流程） |

## 架构关系

```
HunyuanImage3Pipeline (GenerationMixin)
  ├── TokenizerWrapper (多模态序列构建)
  ├── HunyuanImage3ImageProcessor (图像尺寸处理)
  ├── Siglip2VisionModel (条件图像 ViT 编码)
  ├── LightProjector (ViT -> Transformer 维度对齐)
  ├── AutoencoderKLConv3D (VAE 编解码)
  │     ├── Encoder (DCAE 下采样)
  │     └── Decoder (DCAE 上采样)
  ├── UNetDown / patch_embed (VAE latent -> Transformer hidden)
  ├── UNetUp / final_layer (Transformer hidden -> VAE latent)
  ├── TimestepEmbedder x3 (时间步嵌入)
  ├── HunyuanImage3Model (Decoder-only Transformer)
  │     ├── VocabParallelEmbedding
  │     └── HunyuanImage3DecoderLayer x N
  │           ├── HunYuanAttention (QKV + 2D RoPE)
  │           └── HunYuanSparseMoeBlock / HunYuanMLP
  └── HunyuanImage3Text2ImagePipeline (扩散去噪循环)
        ├── FlowMatchEulerDiscreteScheduler
        └── ClassifierFreeGuidance
```
