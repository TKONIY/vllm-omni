# `flux_transformer.py` -- FLUX.1 Transformer 模型

## 文件概述

本文件实现了 FLUX.1 扩散模型的 Transformer 架构。FLUX.1 采用创新的双流（dual-stream）+ 单流（single-stream）设计：双流块中文本和图像各自维护独立的隐藏状态，通过 Joint Attention 交互；单流块将二者合并处理。所有线性层均使用 vLLM 的张量并行实现。

**文件路径**: `vllm_omni/diffusion/models/flux/flux_transformer.py`

## 关键代码解析

### FluxAttention 联合注意力

```python
class FluxAttention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, added_kv_proj_dim=None, ...):
        self.to_qkv = QKVParallelLinear(...)      # 图像流 QKV
        self.add_kv_proj = QKVParallelLinear(...)  # 文本流 QKV（cross-attention）
        self.norm_q = RMSNorm(dim_head)            # QK 归一化
        self.norm_k = RMSNorm(dim_head)
        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(...)                  # 优化注意力后端
```

双流块中，图像和文本的 QKV 被拼接后送入同一个注意力层计算 Joint Attention，结果再按文本长度分割回两个流。

### FluxTransformerBlock 双流块

```python
class FluxTransformerBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        # 1. AdaLayerNormZero 调制（图像流和文本流各自独立）
        norm_hidden_states, gate_msa, ... = self.norm1(hidden_states, emb=temb)
        norm_encoder, c_gate_msa, ... = self.norm1_context(encoder_hidden_states, emb=temb)
        # 2. Joint Attention
        attn_output, context_attn_output = self.attn(norm_hidden_states, norm_encoder, ...)
        # 3. 各自的残差和 FFN
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
```

### FluxSingleTransformerBlock 单流块

```python
class FluxSingleTransformerBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb, ...):
        # 合并文本和图像
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # 并行计算 attention 和 MLP
        attn_output = self.attn(hidden_states=norm_hidden_states, ...)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        # 合并输出
        hidden_states = gate * self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
```

单流块将文本和图像 token 合并为一个序列处理，同时并行计算注意力和 MLP。

### FluxPosEmbed 多轴旋转位置编码

```python
class FluxPosEmbed(nn.Module):
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(self.axes_dim[i], pos[:, i], ...)
```

FLUX 使用三轴 RoPE（时间轴 16 维 + 高度 56 维 + 宽度 56 维 = 128 维），分别为文本和图像 token 生成位置编码。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FluxTransformer2DModel` | nn.Module | FLUX.1 完整 Transformer |
| `FluxTransformerBlock` | nn.Module | 双流 Joint Attention 块 |
| `FluxSingleTransformerBlock` | nn.Module | 单流 Transformer 块 |
| `FluxAttention` | nn.Module | 联合注意力（支持 cross-attention） |
| `FluxPosEmbed` | nn.Module | 多轴 RoPE 位置编码 |
| `FeedForward` | nn.Module | GELU 前馈网络 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_flux.py` | Pipeline 创建并调用 Transformer |
| 依赖 | vLLM 并行层 | QKVParallelLinear、RowParallelLinear 等 |
| 依赖 | `vllm_omni.diffusion.attention` | 优化注意力后端 |
| 依赖 | `vllm_omni.diffusion.layers.rope` | RotaryEmbedding |

## 总结

`FluxTransformer2DModel` 实现了 FLUX.1 的双流 + 单流 Transformer 架构。双流块让文本和图像保持独立表示并通过 Joint Attention 交互（19 层默认），单流块将二者合并处理（38 层默认）。所有投影层使用 vLLM 张量并行层，配合 QK RMSNorm 和多轴 RoPE，实现高效的多 GPU 推理。
