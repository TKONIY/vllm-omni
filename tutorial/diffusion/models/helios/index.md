# Helios 模型模块索引

## 概述

Helios 是一个基于 Wan2.2 架构扩展的分块长视频生成模型，支持文本到视频（T2V）、图像到视频（I2V）和视频到视频（V2V）。其核心创新包括多期记忆补丁、引导交叉注意力和金字塔多阶段去噪。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块初始化，统一导出核心组件 |
| [`helios_transformer.py`](helios_transformer.py.md) | 3D Transformer 核心模型，包含多期记忆补丁、历史放大等机制 |
| [`pipeline_helios.py`](pipeline_helios.py.md) | 完整推理管线，分块去噪 + T2V/I2V/V2V 支持 |
| [`scheduling_helios.py`](scheduling_helios.py.md) | 统一调度器，集成 Euler/UniPC/DMD 三种算法 |

## 架构关系

```
HeliosPipeline
  ├── UMT5EncoderModel (文本编码器)
  ├── AutoencoderKLWan (VAE 编码/解码)
  ├── HeliosTransformer3DModel (核心 Transformer)
  │     ├── HeliosRotaryPosEmbed (3D 旋转位置编码)
  │     ├── HeliosTimeTextEmbedding (时间步+文本嵌入)
  │     ├── Multi-term Memory Patches (短/中/长期历史编码)
  │     └── HeliosTransformerBlock x N
  │           ├── HeliosSelfAttention (含历史放大)
  │           ├── HeliosCrossAttention (引导交叉注意力)
  │           └── HeliosFeedForward
  └── HeliosScheduler (Euler / UniPC / DMD)
```
