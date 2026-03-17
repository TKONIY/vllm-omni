# `bagel_transformer.py` -- Bagel MoT Transformer 核心模型

## 文件概述

本文件实现了 Bagel 模型的核心 Transformer 架构，基于 Qwen2 的 Mixture-of-Tokens (MoT) 变体。MoT 架构将理解（understanding）和生成（generation）任务分流到不同的注意力头和 MLP 路径上，使单一模型可同时处理多模态理解和图像生成。代码大量使用 vLLM 的张量并行层（QKVParallelLinear、RowParallelLinear 等）。

**文件路径**: `vllm_omni/diffusion/models/bagel/bagel_transformer.py`

## 关键代码解析

### Qwen2MoTConfig 配置

```python
class Qwen2MoTConfig(Qwen2Config):
    model_type = "qwen2_mot"
    def __init__(self, ..., layer_module="Qwen2MoTDecoderLayer", ...):
        self.qk_norm = qk_norm
        self.layer_module = layer_module
```

继承 Qwen2 配置，增加 `qk_norm` 和 `layer_module` 参数。`layer_module` 决定使用 MoT 层还是普通层。

### NaiveCache KV 缓存

```python
class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}
```

简单的字典式 KV 缓存，按层索引存储键值张量，用于自回归式的图像生成。

### PackedAttentionMoT 混合注意力

```python
class PackedAttentionMoT(nn.Module):
    def __init__(self, config, layer_idx):
        # 理解模式投影 (stacked q/k/v)
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        # 生成模式 MoE 投影 (stacked q/k/v)
        self.qkv_proj_moe_gen = QKVParallelLinear(...)
        self.o_proj_moe_gen = RowParallelLinear(...)
```

核心创新点：为理解和生成任务分别维护独立的 QKV 投影权重。通过 `mode` 参数在 `"und"`（理解）和 `"gen"`（生成）之间切换。使用 vLLM 的 `flash_attn_varlen_func` 实现高效的变长序列注意力。

### Bagel 主类

`Bagel` 类是模型的最高层封装，提供以下核心方法：

1. **`prepare_prompts()`**: 将文本 prompt 编码为 token ids 和位置编码
2. **`prepare_vae_images()`**: 准备 VAE 图像输入（用于图生图）
3. **`prepare_vit_images()`**: 准备 ViT 图像输入（用于图像理解）
4. **`prepare_vae_latent()`**: 准备 VAE 潜空间的初始噪声
5. **`forward_cache_update_text()`**: 文本 prefill 并更新 KV 缓存
6. **`forward_cache_update_vae()`**: VAE 图像 prefill 并更新 KV 缓存
7. **`generate_image()`**: 执行完整的去噪循环生成图像

### patchify 辅助函数

```python
def patchify(imgs, p):
    x = imgs.reshape(imgs.shape[0], 3, imgs.shape[2] // p, p, imgs.shape[3] // p, p)
    x = torch.einsum("nchpwq->nhwcpq", x)
    x = x.reshape(imgs.shape[0], -1, 3 * p**2)
```

将图像分割为 patch 序列，使用 `einsum` 进行高效的维度重排。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Bagel` | nn.Module | 模型最高层封装，管理 LLM + VAE + ViT |
| `Qwen2MoTForCausalLM` | nn.Module | MoT 因果语言模型 |
| `Qwen2MoTConfig` | Config | MoT 模型配置 |
| `PackedAttentionMoT` | nn.Module | 双路注意力（理解 + 生成） |
| `NaiveCache` | 数据类 | 简单 KV 缓存 |
| `MLPconnector` | nn.Module | ViT 到 LLM 的连接器 |
| `BagelRotaryEmbedding` | nn.Module | 独立的旋转位置编码 |
| `BagelMLP` | nn.Module | 门控 MLP（SiLU 激活） |
| `patchify()` | 函数 | 图像 patch 化 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_bagel.py` | Pipeline 创建并调用 Bagel 实例 |
| 依赖 | `autoencoder.py` | 使用 AutoEncoder 进行图像编解码 |
| 依赖 | vLLM 并行层 | 使用 QKVParallelLinear 等实现张量并行 |
| 依赖 | `flash_attn_varlen_func` | 变长序列闪存注意力 |

## 总结

`bagel_transformer.py` 是 Bagel 模型的核心，实现了 Mixture-of-Tokens 架构。其关键设计是通过双路 QKV 投影和 MLP 实现理解/生成任务的分流，同时利用 vLLM 的高效并行层支持多 GPU 推理。模型支持文本 prefill、图像 prefill（VAE + ViT）、以及迭代式图像生成等多种操作模式。
