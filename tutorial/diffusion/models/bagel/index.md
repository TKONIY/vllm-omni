# bagel/ -- Bagel 模型目录索引

## 目录概述

Bagel 是一个基于 Mixture-of-Tokens (MoT) 架构的多模态生成模型，将理解和生成任务统一在单一 Transformer 中。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化（空） |
| [`autoencoder.py`](autoencoder.md) | VAE 自编码器（编码/解码图像潜变量） |
| [`bagel_transformer.py`](bagel_transformer.md) | MoT Transformer 核心（双路注意力） |
| [`pipeline_bagel.py`](pipeline_bagel.md) | 完整推理管线（LLM + ViT + VAE + 双重 CFG） |

## 架构概览

```
BagelPipeline
  |-- Bagel (核心模型)
  |     |-- Qwen2MoTForCausalLM (MoT 语言模型)
  |     |     |-- PackedAttentionMoT (双路 QKV: und/gen)
  |     |     |-- BagelMLP (门控 SiLU MLP)
  |     |-- SiglipNaViTWrapper (SigLIP ViT)
  |     |-- latent_pos_embed, time_embedder, ...
  |-- AutoEncoder (VAE)
  |-- tokenizer, image_processor
```

## 核心特色

1. **MoT 架构**: 理解和生成使用独立的 QKV/MLP 权重，共享 embedding 和位置编码
2. **双重 CFG**: 同时对文本条件和图像条件进行无条件引导（三组 KV 缓存）
3. **NaViT 图像处理**: 支持可变分辨率的 packed 图像输入
4. **自包含管线**: Pipeline 内嵌 LLM 和 ViT，无需外部 AR 阶段

## 总结

Bagel 模型的独特之处在于将多模态理解和生成统一在单一 MoT Transformer 中。双路注意力机制让同一模型可以处理图像理解（ViT 编码）和图像生成（VAE 解码）任务。
