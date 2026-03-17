# `mammothmoda2_dit_model.py` — MammothModa2 DiT 模型

## 文件概述

该文件实现了 MammothModa2 的 DiT (Diffusion Transformer) 模型，采用 Lumina2 架构风格。模型包含三个 Refiner 模块（context、noise、reference image），主 Transformer 层以及可选的 Q-Former 图像条件精化器。使用 Qwen2 的 RMSNorm 和 Flash Attention 优化。

## 关键代码解析

### 自适应层归一化 — LuminaRMSNormZero

```python
class LuminaRMSNormZero(nn.Module):
    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp
```

生成 4 个调制参数：MSA 缩放/门控、MLP 缩放/门控。

### 注意力处理器 — AttnProcessor

```python
class AttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, ...):
        # 应用 RoPE
        query = apply_real_rotary_emb(query, image_rotary_emb[0], image_rotary_emb[1])
        # 支持 Flash Attention varlen 和 PyTorch SDPA 两种路径
        if _HAS_FLASH_ATTN_VARLEN and attention_mask is not None:
            attn_output = flash_attn_varlen_func(...)
        else:
            hidden_states = F.scaled_dot_product_attention(...)
```

同时支持 Flash Attention 变长序列和标准 SDPA 注意力。

### Q-Former 图像条件精化器

```python
class SimpleQFormerImageRefiner(nn.Module):
    def __init__(self, hidden_size, num_queries=128, num_layers=2):
        self.query = nn.Parameter(scale * torch.randn(1, num_queries, hidden_size))
        # 每层包含：自注意力 -> 交叉注意力 -> FFN
```

使用可学习 query 从图像条件 token 中提取固定数量的特征表示。

### 主 Transformer — Transformer2DModel

```python
class Transformer2DModel(ModelMixin, ConfigMixin):
    def __init__(self, ...):
        self.noise_refiner = nn.ModuleList([...])      # 噪声 latent 精化
        self.ref_image_refiner = nn.ModuleList([...])  # 参考图像精化
        self.context_refiner = nn.ModuleList([...])    # 文本条件精化（无调制）
        self.layers = nn.ModuleList([...])             # 主 Transformer 层
```

分阶段处理：先分别精化各模态表示，再拼接进行联合 Transformer 处理。

### 前向传播流程

```python
def forward(self, hidden_states, timestep, text_hidden_states, freqs_cis, text_attention_mask, ...):
    # 1. 准备嵌入（patchify + embeddings + RoPE）
    # 2. 分别精化文本和噪声
    text_hidden_states, img_tokens = self._apply_refiners(...)
    # 3. 拼接为联合序列
    joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i]
    joint_hidden_states[i, encoder_seq_len:encoder_seq_len + img_len] = img_tokens[i]
    # 4. 主 Transformer 处理
    hidden_states = self._apply_transformer_layers(joint_hidden_states, ...)
    # 5. 提取图像部分并 unpatchify
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Transformer2DModel` | 类 | 主 DiT Transformer 模型 |
| `TransformerBlock` | 类 | Transformer 块（支持可选调制） |
| `LuminaRMSNormZero` | 类 | 自适应 RMSNorm 零初始化 |
| `LuminaFeedForward` | 类 | SwiGLU 前馈网络 |
| `LuminaLayerNormContinuous` | 类 | 连续自适应层归一化 |
| `Lumina2CombinedTimestepCaptionEmbedding` | 类 | 联合时间步和文本嵌入 |
| `SimpleQFormerImageRefiner` | 类 | Q-Former 图像条件精化器 |
| `AttnProcessor` | 类 | 注意力处理器（Flash Attn/SDPA） |

## 与其他模块的关系

- 使用 `rope_real.py` 中的旋转位置编码实现
- 被 `pipeline_mammothmoda2_dit.py` 引用作为核心去噪模型
- 使用 diffusers 的 `ConfigMixin`/`ModelMixin` 实现配置化
- 使用 Qwen2RMSNorm 替代默认归一化层

## 总结

MammothModa2 DiT 模型采用分阶段处理架构：先通过 Refiner 分别精化文本、噪声和参考图像的表示，再拼接进行联合 Transformer 处理。模型使用 Lumina2 风格的自适应归一化、SwiGLU FFN 和可选的 Q-Former 图像条件精化器，支持灵活的多模态条件控制。
