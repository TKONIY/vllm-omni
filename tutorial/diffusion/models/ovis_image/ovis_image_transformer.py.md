# `ovis_image_transformer.py` — Ovis Image Transformer 模型

## 文件概述

该文件实现了 Ovis Image 的 2D Transformer 模型，基于 Flux 架构，采用双流/单流混合设计。与 LongCat 类似但有几个关键差异：使用 SwiGLU FFN、不使用序列并行、使用 `RotaryEmbedding` 而非 `apply_rotary_emb`、单流块的 MLP 使用 SiLU 门控。

## 关键代码解析

### OvisImageAttention

```python
class OvisImageAttention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, ...):
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim, head_size=self.head_dim,
            total_num_heads=self.heads,
            disable_tp=True,  # 禁用张量并行
            bias=bias,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(num_heads=heads, head_size=self.head_dim, ...)
```

使用 `RotaryEmbedding` 类直接应用旋转位置编码，而非函数式 `apply_rotary_emb`。

### 单流块 — SiLU 门控 MLP

```python
class OvisImageSingleTransformerBlock(nn.Module):
    def __init__(self, dim, ...):
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim * 2)  # 输出 2x 维度
        self.act_mlp = nn.SiLU()

    def forward(self, hidden_states, ...):
        mlp_hidden_states, mlp_hidden_gate = torch.split(
            self.proj_mlp(norm_hidden_states), [self.mlp_hidden_dim, self.mlp_hidden_dim], dim=-1
        )
        mlp_hidden_states = self.act_mlp(mlp_hidden_gate) * mlp_hidden_states
```

使用 SiLU 门控的 MLP，投影到 2 倍隐藏维度后分割为值和门控。

### 双流块 — SwiGLU FFN

```python
class OvisImageTransformerBlock(nn.Module):
    def __init__(self, dim, ...):
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="swiglu")
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="swiglu")
```

使用 diffusers 的 `FeedForward` 配合 SwiGLU 激活函数。

### 位置编码

```python
class OvisImagePosEmbed(nn.Module):
    def forward(self, ids):
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i], theta=self.theta,
                use_real=False,  # 使用复数形式
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
```

注意使用 `use_real=False`（复数形式），与 LongCat 的 `use_real=True` 不同。

### 主模型

```python
class OvisImageTransformer2DModel(nn.Module):
    def __init__(self, od_config, ...):
        self.context_embedder_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
```

文本嵌入先通过 RMSNorm 归一化再投影，这是 Ovis 特有的设计。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OvisImageTransformer2DModel` | 类 | 主 Transformer 模型 |
| `OvisImageAttention` | 类 | 注意力层（禁用 TP） |
| `OvisImageTransformerBlock` | 类 | 双流块（SwiGLU FFN） |
| `OvisImageSingleTransformerBlock` | 类 | 单流块（SiLU 门控 MLP） |
| `OvisImagePosEmbed` | 类 | 多轴位置编码（复数形式） |

## 与其他模块的关系

- 被 `pipeline_ovis_image.py` 使用
- 使用 `vllm_omni.diffusion.layers.rope.RotaryEmbedding` 应用 RoPE
- 使用 `vllm_omni.diffusion.attention.layer.Attention` 进行注意力计算

## 总结

Ovis Image Transformer 是一个 Flux 风格的双流/单流混合模型，与 LongCat 的主要差异包括：(1) 禁用张量并行；(2) 使用复数形式的 RoPE；(3) 使用 SwiGLU FFN 和 SiLU 门控 MLP；(4) 文本嵌入额外经过 RMSNorm 归一化。
