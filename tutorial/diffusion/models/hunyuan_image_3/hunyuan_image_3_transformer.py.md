# `hunyuan_image_3_transformer.py` — HunyuanImage3 Transformer 模型与管线

## 文件概述

本文件是 HunyuanImage3 模型的核心实现，包含约 2750 行代码，定义了完整的 Decoder-only Transformer 模型、2D RoPE 嵌入器、MoE 块、注意力层、图像处理组件以及文本到图像的扩散管线。HunyuanImage3 采用自回归（AR）架构将图像生成建模为序列预测任务，其中文本和图像 token 共享同一个 Transformer。

## 关键代码解析

### 1. 配置类

```python
class HunyuanImage3Config(PretrainedConfig):
    model_type = "Hunyuan"
    def __init__(self, vocab_size=290943, hidden_size=4096, num_hidden_layers=32,
                 num_experts=1, moe_topk=1, vae_downsample_factor=(16, 16),
                 img_proj_type="unet", patch_size=1, ...):
```

非常丰富的配置类，支持 MoE、MLA（Multi-Latent Attention）、CLA（Cross-Layer Attention）等多种架构变体。

### 2. 2D RoPE 嵌入

```python
class HunYuanRotary2DEmbedder:
    def __call__(self, q, k, hidden_states, custom_pos_emb, **kwargs):
        if kwargs.get("mode", "gen_text") != "gen_image":
            # 文本模式：使用标准 1D RoPE
            ...
        else:
            # 图像模式：使用 2D RoPE
            ...
```

在文本生成模式下使用标准 1D RoPE，在图像生成模式下使用 2D RoPE 以保留图像的空间结构信息。

### 3. MoE 稀疏块

```python
class HunYuanSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx, prefix=""):
        self.gate = ReplicatedLinear(config.hidden_size, num_experts, ...)
        self.experts = HunyuanFusedMoE(
            num_experts=num_experts, top_k=moe_topk,
            hidden_size=config.hidden_size,
            intermediate_size=moe_intermediate_size, ...
        )
        self.shared_experts = HunYuanMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_intermediate_size, ...
        )
```

每层包含路由专家（通过 `HunyuanFusedMoE` 实现）和共享专家（标准 MLP），使用门控网络选择 top-k 专家。

### 4. 注意力层

```python
class HunYuanAttention(nn.Module):
    def __init__(self, config, layer_idx, prefix=""):
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        self.rotary_emb_fn = HunYuanRotary2DEmbedder(num_heads, num_kv_heads, head_dim)
        self.attn = Attention(num_heads=self.num_heads, head_size=head_dim, ...)
```

使用 vLLM 的 `QKVParallelLinear` 实现张量并行，通过统一 `Attention` 层进行高效注意力计算。

### 5. HunyuanImage3Model

```python
class HunyuanImage3Model(nn.Module):
    def __init__(self, config, prefix=""):
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size, ...)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HunyuanImage3DecoderLayer(config=config, ...),
        )
        self.norm = RMSNorm(config.hidden_size, ...)
```

标准的 Decoder-only 架构，支持流水线并行（PP）和张量并行（TP）。

### 6. ClassifierFreeGuidance

```python
class ClassifierFreeGuidance:
    def __call__(self, pred_cond, pred_uncond, guidance_scale, step):
        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift
        return pred
```

### 7. UNet 风格图像投影

```python
class UNetDown(nn.Module):  # patch_embed
    # 将 VAE 潜在空间映射到 Transformer 隐藏空间
    # 包含 ResBlock + 时间步条件

class UNetUp(nn.Module):   # final_layer
    # 将 Transformer 隐藏状态映射回 VAE 潜在空间
    # 包含反向 ResBlock + 时间步条件
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HunyuanImage3Config` | 类 | 模型配置（支持 MoE/MLA/CLA） |
| `Resolution` / `ResolutionGroup` | 类 | 图像分辨率管理 |
| `ImageInfo` / `JointImageInfo` | 数据类 | 图像元信息 |
| `LightProjector` | 类 | ViT 特征投影器 |
| `HunYuanRotary2DEmbedder` | 类 | 2D RoPE 嵌入器 |
| `HunYuanMLP` | 类 | 标准 MLP（门控 SiLU） |
| `HunYuanSparseMoeBlock` | 类 | MoE 稀疏专家块 |
| `HunYuanAttention` | 类 | 注意力层（TP 并行） |
| `HunyuanImage3DecoderLayer` | 类 | Decoder 层 |
| `HunyuanImage3Model` | 类 | 核心 Transformer 模型 |
| `ClassifierFreeGuidance` | 类 | CFG 引导 |
| `HunyuanImage3Text2ImagePipeline` | 类 | 文本到图像扩散管线 |
| `TimestepEmbedder` | 类 | 正弦时间步嵌入 |
| `UNetDown` / `UNetUp` | 类 | 图像投影/反投影 |

## 与其他模块的关系

- **`pipeline_hunyuan_image_3.py`**：`HunyuanImage3Pipeline` 封装本文件的模型和管线
- **`hunyuan_fused_moe.py`**：MoE 层实现
- **`autoencoder.py`**：VAE 编解码
- **`hunyuan_image_3_tokenizer.py`**：序列构建
- **`vllm_omni.diffusion.attention.layer`**：统一注意力层
- **vLLM layers**：张量并行基础层

## 总结

`hunyuan_image_3_transformer.py` 是 HunyuanImage3 的核心大文件，实现了完整的 Decoder-only Transformer 架构。模型将文本和图像 token 统一处理，通过 2D RoPE 保留空间信息，MoE 提供大容量专家能力，UNet 风格的投影层完成潜在空间与隐藏空间的转换。扩散去噪环节在 `HunyuanImage3Text2ImagePipeline` 中完成。
