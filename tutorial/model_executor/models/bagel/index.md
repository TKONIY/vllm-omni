# Bagel 模型模块架构概览

## 模块简介

Bagel 模型是一个支持图像理解和图像生成（img2img）的多模态条件生成模型。它基于 vLLM 原生的 `BagelForConditionalGeneration` 进行扩展，增加了 VAE 编码器和 MoT（Mixture-of-Transformers）路由机制，使 AR 阶段能够同时处理 VAE 潜变量和 ViT 视觉特征。

## 架构图

```
用户请求 (文本 + 图像)
       │
       ▼
┌──────────────────────────────┐
│  OmniBagelMultiModalProcessor│  ← 多模态预处理（图像/img2img）
│  OmniBagelProcessingInfo     │  ← 配置信息与数据解析
│  OmniBagelDummyInputsBuilder │  ← 性能分析用虚拟输入
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  OmniBagelForConditionalGeneration       │
│  ├── VAEEncoder (vae)                    │  ← 图像编码为潜变量
│  │   ├── Encoder                         │
│  │   └── DiagonalGaussian                │
│  ├── vae2llm (Linear)                    │  ← 潜变量投影到LLM维度
│  ├── latent_pos_embed (PositionEmbedding)│  ← 潜变量位置编码
│  ├── time_embedder (TimestepEmbedder)    │  ← 时间步嵌入
│  ├── MoT 权重模块 (*_moe_gen)            │  ← 每层的生成模式权重
│  └── BagelForConditionalGeneration (基类) │  ← Qwen2 LLM + ViT
└──────────────────────────────────────────┘
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 模块入口，导出 `OmniBagelForConditionalGeneration` |
| `bagel.py` | 核心实现：多模态处理器、VAE 编码器、MoT 前向传播、权重加载 |

## 核心设计思想

1. **MoT 路由机制**：VAE 潜变量 token 走专用的 `*_moe_gen` 权重矩阵（QKV、O、MLP），其余 token（ViT、文本）走标准理解模式权重，确保 KV cache 与 DiT 阶段兼容。

2. **位置编码对齐**：img2img 模式下，VAE token 统一使用 position 0，ViT token 使用 position 1，文本 token 从 position 2 开始递增，与单阶段 DiT 管线的位置方案保持一致。

3. **KV 元数据传递**：通过 `flush_pending_metadata` 和 `get_kv_transfer_metadata` 机制将 rope 偏移量和图像尺寸传递给下游 DiT 阶段。
