# diffusion/models/ -- 扩散模型实现目录索引

## 目录概述

`diffusion/models/` 是 vllm-omni 项目中扩散模型的核心实现目录。包含各种文本到图像、文本到音频扩散模型的 Transformer 架构和推理管线，全部使用 vLLM 的高效推理基础设施（张量并行、优化注意力后端等）。

## 根文件

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化 |
| [`interface.py`](interface.md) | 模型能力接口协议（图像输入/音频输入/音频输出） |
| [`progress_bar.py`](progress_bar.md) | 分布式友好的进度条混入类 |

## 子目录

### 基础设施

| 目录 | 说明 | 文档 |
|------|------|------|
| [`schedulers/`](schedulers/index.md) | 噪声调度器（FlowUniPC 等） |

### 图像生成模型

| 目录 | 模型 | 架构特色 | 文档 |
|------|------|----------|------|
| [`flux/`](flux/index.md) | FLUX.1-dev | 双流+单流 Transformer, CLIP+T5 | [详情](flux/index.md) |
| [`flux2/`](flux2/index.md) | Flux 2 | SwiGLU, 全局 Modulation, Mistral3 编码器 | [详情](flux2/index.md) |
| [`flux2_klein/`](flux2_klein/index.md) | Flux 2 Klein | Flux 2 + 序列并行 (Ulysses/Ring) | [详情](flux2_klein/index.md) |
| [`sd3/`](sd3/index.md) | Stable Diffusion 3 | MMDiT, 3x 文本编码器, Dual Attention | [详情](sd3/index.md) |
| [`bagel/`](bagel/index.md) | Bagel | MoT 架构, 双重 CFG, SigLIP ViT | [详情](bagel/index.md) |
| [`glm_image/`](glm_image/index.md) | GLM-Image | 两阶段 (AR+DiT), KV 缓存, 字形嵌入 | [详情](glm_image/index.md) |

### 音频生成模型

| 目录 | 模型 | 架构特色 | 文档 |
|------|------|----------|------|
| [`stable_audio/`](stable_audio/index.md) | Stable Audio Open | 1D DiT, GQA, SwiGLU, 时长条件 | [详情](stable_audio/index.md) |
| [`cosyvoice3_audio/`](cosyvoice3_audio/index.md) | CosyVoice3 | 语音合成 DiT, 因果卷积, 说话人嵌入 | [详情](cosyvoice3_audio/index.md) |

## 架构对比

### Transformer 块设计

| 模型 | 注意力类型 | 位置编码 | 激活函数 | 调制方式 |
|------|-----------|---------|---------|---------|
| FLUX.1 | Joint Attention + Single Stream | 3轴 RoPE (16+56+56) | GELU | 逐块 AdaLNZero |
| Flux 2 | Joint Attention + Parallel MLP | 4轴 RoPE (32x4) | SwiGLU | 全局 Modulation |
| SD3 | Joint Attention (MMDiT) | 学习位置编码 | GELU | 逐块 AdaLNZero |
| Bagel | MoT 双路注意力 | Qwen2 RoPE | SiLU | 无 |
| GLM-Image | Joint Attention + KV Cache | 2D RoPE | GELU | 12参数 AdaLN |
| Stable Audio | Self + Cross Attn | 部分 RoPE | SwiGLU | 无 |
| CosyVoice3 | Self Attention | RotaryEmbedding | GELU | AdaLNZero |

### 文本编码器配置

| 模型 | 编码器 | 特色 |
|------|--------|------|
| FLUX.1 | CLIP + T5 | Pooled (CLIP) + Sequence (T5) |
| Flux 2 | Mistral3 | 多模态（文本+图像输入） |
| Flux 2 Klein | Qwen3 | 隐藏状态作为嵌入 |
| SD3 | CLIP x2 + T5 | 三编码器拼接 |
| Bagel | Qwen2 Tokenizer | MoT 内部处理 |
| GLM-Image | ByT5 | 字形级编码 |
| Stable Audio | T5 | 音频条件投影 |

## 公共设计模式

1. **Pipeline 结构**: 所有 Pipeline 遵循 `__init__` -> `encode_prompt` -> `prepare_latents` -> `diffuse` -> `decode` 流程
2. **权重加载**: 统一使用 `load_weights()` 方法 + `AutoWeightsLoader`，处理 QKV 融合映射
3. **CFG 并行**: 通过 `CFGParallelMixin` 实现多 GPU CFG 加速
4. **请求接口**: 统一接受 `OmniDiffusionRequest`，返回 `DiffusionOutput`
5. **张量并行**: Transformer 层使用 vLLM 的 `QKVParallelLinear`、`RowParallelLinear` 等
