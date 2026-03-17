# `cosyvoice3_dit.py` -- CosyVoice3 扩散 Transformer (DiT)

## 文件概述

本文件实现了 CosyVoice3 的扩散 Transformer 模型，用于语音合成的去噪过程。该模型将文本条件（mu）、条件音频（cond）和噪声音频（x）融合，通过多层 DiT 块进行迭代去噪。模型已重构为使用 vllm-omni 的优化注意力后端（FlashAttention/SageAttention/SDPA），取代了原始的注意力实现。

**文件路径**: `vllm_omni/diffusion/models/cosyvoice3_audio/cosyvoice3_dit.py`

## 关键代码解析

### DiTAttention 优化注意力

```python
class DiTAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        # Q/K/V 独立线性投影
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        # 使用 vllm-omni 扩散注意力后端
        self.attn = DiffusionAttention(
            num_heads=heads, head_size=dim_head,
            softmax_scale=self.scale, causal=False,
        )
```

用 vllm-omni 的 `DiffusionAttention` 替代原始注意力，自动选择最优后端。

### DiTBlock 带 AdaLayerNorm 调制的 Transformer 块

```python
class DiTBlock(nn.Module):
    def forward(self, x, t, mask=None, rope=None):
        # AdaLayerNormZero 进行时间步条件调制
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        # FFN with modulation
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(ff_norm)
```

使用 AdaLayerNormZero 进行时间步条件调制，通过 gate/scale/shift 参数控制注意力和 FFN 的输出。

### InputEmbedding 多模态输入融合

```python
class InputEmbedding(nn.Module):
    def forward(self, x, cond, text_embed, spks):
        to_cat = [x, cond, text_embed]   # 噪声音频 + 条件音频 + 文本
        if self.spk_dim > 0:
            to_cat.append(spks)           # + 说话人嵌入
        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x   # 因果卷积位置编码
```

将多个条件信号拼接后投影到统一维度，并添加因果卷积位置编码。

### CausalConvPositionEmbedding 因果卷积位置编码

```python
class CausalConvPositionEmbedding(nn.Module):
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))  # 左填充实现因果性
        x = self.conv1(x)
```

使用左填充确保因果性（不依赖未来信息），适用于流式语音合成。

### DiT 主模型

```python
class DiT(nn.Module):
    def forward(self, x, mask, mu, t, spks=None, cond=None):
        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        output = self.proj_out(x).transpose(1, 2)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiT` | nn.Module | DiT 主模型，包含完整 Transformer 管线 |
| `DiTBlock` | nn.Module | DiT Transformer 块（AdaLN + Attention + FFN） |
| `DiTAttention` | nn.Module | 使用 vllm-omni 后端的注意力层 |
| `InputEmbedding` | nn.Module | 多模态输入融合模块 |
| `TimestepEmbedding` | nn.Module | 时间步嵌入（正弦 + MLP） |
| `TextEmbedding` | nn.Module | 文本嵌入（可选 ConvNeXt 建模） |
| `CausalConvPositionEmbedding` | nn.Module | 因果卷积位置编码 |
| `FeedForward` | nn.Module | GELU 前馈网络 |
| `AdaLayerNormZero_Final` | nn.Module | 最终层的自适应层归一化 |
| `ConvNeXtV2Block` | nn.Module | ConvNeXt-V2 块（用于文本建模） |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `vllm_omni.diffusion.attention.layer` | 使用优化的 `Attention` 后端 |
| 依赖 | `x_transformers` | 使用 `RotaryEmbedding` 和 `apply_rotary_pos_emb` |
| 依赖 | `diffusers.models.normalization` | 使用 `AdaLayerNormZero` |
| 来源 | CosyVoice / FunAudioLLM | 改编自阿里的 CosyVoice 项目 |

## 总结

CosyVoice3 DiT 是一个面向语音合成的扩散 Transformer。其核心设计包括：(1) 多模态输入融合（噪声音频、条件音频、文本、说话人嵌入），(2) 因果卷积位置编码确保流式推理能力，(3) AdaLayerNormZero 时间步调制，(4) 可选的长跳跃连接（类 U-Net 残差）。通过接入 vllm-omni 的注意力后端，实现了对 FlashAttention 等高效后端的透明支持。
