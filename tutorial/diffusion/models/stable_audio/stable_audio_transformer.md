# `stable_audio_transformer.py` -- Stable Audio DiT 模型

## 文件概述

实现 Stable Audio Open 的 DiT 模型，用于文本到音频生成。该模型处理 1D 音频潜变量（而非 2D 图像潜变量），使用自注意力 + 交叉注意力 + SwiGLU FFN 的标准 Transformer 块。支持 GQA（分组查询注意力）以提升交叉注意力效率。

**文件路径**: `vllm_omni/diffusion/models/stable_audio/stable_audio_transformer.py`

## 关键代码解析

### 模型架构流程

```
输入: [B, 64, L] (音频潜变量)
  -> preprocess_conv (残差 1D 卷积)
  -> proj_in (64 -> 1536, inner_dim)
  -> 前置 global_hidden_states (时间+时长条件)
  -> 24 层 DiTBlock (自注意力 + 交叉注意力 + FFN)
  -> proj_out (1536 -> 64)
  -> postprocess_conv (残差 1D 卷积)
输出: [B, 64, L]
```

### StableAudioGaussianFourierProjection 时间步嵌入

```python
class StableAudioGaussianFourierProjection(nn.Module):
    def forward(self, x):
        x_proj = 2 * math.pi * x[:, None] @ self.weight[None, :]
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
```

使用高斯傅里叶特征对时间步进行嵌入（flip_sin_to_cos=True）。

### GQA 交叉注意力

```python
class StableAudioCrossAttention(nn.Module):
    def __init__(self, ...):
        self.to_q = ReplicatedLinear(dim, inner_dim, ...)   # 全头 Q
        self.to_k = ReplicatedLinear(cross_dim, kv_dim, ...)  # 少头 K
        self.to_v = ReplicatedLinear(cross_dim, kv_dim, ...)  # 少头 V
    def forward(self, ...):
        # 手动扩展 KV 头匹配 Q 头
        key = key.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
        key = key.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)
```

交叉注意力使用 GQA：Q 使用全部注意力头，KV 使用较少的头并手动扩展。

### apply_rotary_emb_stable_audio 部分 RoPE

```python
def apply_rotary_emb_stable_audio(hidden_states, freqs_cis):
    rotary_dim = cos.shape[-1]
    x_rot = hidden_states[..., :rotary_dim]    # 前半部分应用 RoPE
    x_pass = hidden_states[..., rotary_dim:]   # 后半部分不变
```

仅对 head_dim 的前半部分应用旋转位置编码。

### SwiGLU FFN

```python
class SwiGLU(nn.Module):
    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `StableAudioDiTModel` | nn.Module | 完整 DiT 模型 |
| `StableAudioDiTBlock` | nn.Module | DiT 块（self-attn + cross-attn + FFN） |
| `StableAudioSelfAttention` | nn.Module | 全头自注意力 |
| `StableAudioCrossAttention` | nn.Module | GQA 交叉注意力 |
| `StableAudioGaussianFourierProjection` | nn.Module | 高斯傅里叶时间步嵌入 |
| `SwiGLU` | nn.Module | SwiGLU 激活 |
| `StableAudioFeedForward` | nn.Module | SwiGLU FFN |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_stable_audio.py` | Pipeline 调用 |
| 依赖 | `vllm_omni.diffusion.attention.layer` | 优化注意力后端 |
| 依赖 | vLLM ReplicatedLinear | 非张量并行的线性层 |

## 总结

Stable Audio DiT 是一个面向 1D 音频潜空间的 Transformer，其特色包括：(1) 残差 1D 卷积前/后处理，(2) 全局条件 token（时间+时长）前置到序列，(3) GQA 交叉注意力（24 Q 头 / 12 KV 头），(4) 部分 RoPE（仅 head_dim 前半部分），(5) SwiGLU FFN。
