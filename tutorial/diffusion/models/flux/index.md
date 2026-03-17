# flux/ -- FLUX.1 模型目录索引

## 目录概述

FLUX.1-dev 是 Black Forest Labs 推出的高质量文本到图像扩散模型。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化，导出核心类 |
| [`flux_transformer.py`](flux_transformer.md) | 双流+单流 Transformer 架构 |
| [`pipeline_flux.py`](pipeline_flux.md) | 完整推理管线（CLIP+T5, True CFG） |

## 核心特色

1. **双流+单流架构**: 19 层双流 Joint Attention + 38 层单流 Transformer
2. **CLIP+T5 双编码器**: Pooled (CLIP) + Sequence (T5) 文本表示
3. **Latent Packing**: 2x2 patch 打包为序列 token
4. **3轴 RoPE**: 时间(16) + 高度(56) + 宽度(56) = 128 维
5. **True CFG + CFG 并行**: 支持负向提示词和多 GPU 加速
6. **量化支持**: 通过 `quant_config` 支持模型量化

## 总结

FLUX.1 是 vllm-omni 中最经典的图像生成模型实现，展示了双流 Joint Attention 架构如何让文本和图像在独立表示空间中交互。
