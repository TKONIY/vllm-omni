# LTX2 视频生成模型教程索引

## 模块概述

`ltx2` 模块实现了 LTX2 视频生成模型（由 Lightricks 开发），支持文本到视频（T2V）和图像到视频（I2V）两种生成模式。模型可以同时生成视频和配套音频，使用 Gemma3 作为文本编码器。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块入口，导出核心组件 |
| [`ltx2_transformer.py`](ltx2_transformer.py.md) | 3D 视频 Transformer 模型，支持 TP 和 SP |
| [`pipeline_ltx2.py`](pipeline_ltx2.py.md) | 文本到视频+音频联合生成管线 |
| [`pipeline_ltx2_image2video.py`](pipeline_ltx2_image2video.py.md) | 图像到视频生成管线 |

## 架构特点

- **双模态生成**：同步生成视频和音频
- **3D 位置编码**：使用 RoPE3D 处理帧、高度、宽度三个维度
- **I2V 条件掩码**：通过 conditioning_mask 机制实现首帧条件控制
- **CFG 并行**：支持分布式 Classifier-Free Guidance
- **推理缓存**：继承 CachedTransformer 支持注意力缓存
