# sd3/ -- Stable Diffusion 3 模型目录索引

## 目录概述

Stable Diffusion 3 (SD3) 使用 MMDiT 架构，通过 Joint Attention 让文本和图像在 Transformer 中双向交互。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化 |
| [`sd3_transformer.py`](sd3_transformer.md) | MMDiT Transformer（PatchEmbed, Dual Attention） |
| [`pipeline_sd3.py`](pipeline_sd3.md) | 三编码器 Pipeline（CLIP x2 + T5） |

## 核心特色

1. **MMDiT 架构**: 所有层均为双流 Joint Attention（无单流块）
2. **三编码器**: CLIP x2 (pooled) + T5 (sequence) 提供丰富文本表示
3. **Dual Attention**: SD3.5 变体在部分层中添加二次自注意力
4. **PatchEmbed**: Conv2d 直接将潜变量转为 patch 嵌入
5. **分布式 VAE**: 使用 `DistributedAutoencoderKL` 支持大分辨率

## 总结

SD3 的 MMDiT 架构相较 FLUX 更简洁（无单流块设计），所有层都是双流 Joint Attention。三编码器架构提供了目前最丰富的文本条件表示。
