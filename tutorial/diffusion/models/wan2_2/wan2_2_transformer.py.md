# `wan2_2_transformer.py` — Wan2.2 3D Transformer 模型

## 文件概述

本文件实现了 Wan2.2 的核心 3D Transformer 模型 `WanTransformer3DModel`，用于文本条件的视频/图像去噪。架构采用自注意力 + 交叉注意力 + FFN 的标准 DiT 设计，全面使用 vLLM 的张量并行层，支持序列并行和分布式推理。

## 关键代码解析

### 1. 旋转位置编码

```python
def apply_rotary_emb_wan(hidden_states, freqs_cis):
    # 与 Helios 类似的 3D RoPE 应用
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    ...
```

### 2. 3D 旋转位置编码生成

```python
class WanRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        self.DT, self.DY, self.DX = rope_dim

    def forward(self, num_frames, height, width, device):
        # 生成 (T, Y, X) 三维网格的旋转位置编码
        grid_t = torch.arange(num_frames)
        grid_y = torch.arange(height)
        grid_x = torch.arange(width)
        freqs_cos_t, freqs_sin_t = self.get_frequency(self.freqs_base_t, grid_t)
        ...
        return torch.cat([cos_t, cos_y, cos_x, sin_t, sin_y, sin_x], dim=0)
```

### 3. 自注意力

```python
class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, eps):
        self.to_qkv = QKVParallelLinear(hidden_size=dim, head_size=head_dim, total_num_heads=num_heads)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
        self.to_out = RowParallelLinear(self.inner_dim, dim, ...)
        self.attn = Attention(num_heads=self.num_heads, head_size=head_dim, ...)
```

使用 `QKVParallelLinear` 融合 QKV 投影，`DistributedRMSNorm` 在 TP 场景下正确归一化。

### 4. 交叉注意力

```python
class WanCrossAttention(nn.Module):
    # 支持 T2V（文本到视频）和 I2V（图像到视频）两种模式
    def __init__(self, dim, num_heads, head_dim, has_image_input=False):
        self.to_q = ColumnParallelLinear(dim, self.inner_dim, ...)
        self.to_k = ColumnParallelLinear(dim, self.inner_dim, ...)
        self.to_v = ColumnParallelLinear(dim, self.inner_dim, ...)
        if has_image_input:
            self.to_k_img = ColumnParallelLinear(dim, self.inner_dim, ...)
            self.to_v_img = ColumnParallelLinear(dim, self.inner_dim, ...)
```

当模型支持图像输入时，额外添加图像 KV 投影层。

### 5. Transformer 块

```python
class WanTransformerBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb, ...):
        # 1. 调制参数（scale, shift, gate）
        shift_msa, scale_msa, gate_msa, ... = (self.scale_shift_table + temb).chunk(6, dim=1)
        # 2. 自注意力
        norm_hidden_states = (self.norm1(hidden_states) * (1 + scale_msa) + shift_msa)
        attn_output = self.attn1(norm_hidden_states, rotary_emb)
        hidden_states = hidden_states + attn_output * gate_msa
        # 3. 交叉注意力
        attn_output = self.attn2(hidden_states, encoder_hidden_states)
        # 4. FFN
        ff_output = self.ffn(norm_hidden_states)
```

### 6. 模型主类

```python
class WanTransformer3DModel(nn.Module):
    _sp_plan = {
        "rope": SequenceParallelInput(...),
        "blocks.0": SequenceParallelInput(...),
        "proj_out": SequenceParallelOutput(...),
    }

    def forward(self, hidden_states, timestep, encoder_hidden_states, ...):
        # 1. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        # 2. RoPE
        rotary_emb = self.rope(num_frames, height, width, device)
        # 3. 条件嵌入（时间步 + 文本）
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        # 4. Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        # 5. 输出归一化 + 反 patch
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `apply_rotary_emb_wan` | 函数 | 3D RoPE 应用 |
| `DistributedRMSNorm` | 类 | TP 感知 RMSNorm |
| `WanFeedForward` | 类 | TP 并行 FFN |
| `WanRotaryPosEmbed` | 类 | 3D RoPE 生成器 |
| `WanImageEmbedding` | 类 | 图像条件嵌入 |
| `WanTimeTextImageEmbedding` | 类 | 时间步+文本+图像条件嵌入 |
| `WanSelfAttention` | 类 | 自注意力（QKV 融合） |
| `WanCrossAttention` | 类 | 交叉注意力（T2V/I2V） |
| `WanTransformerBlock` | 类 | Transformer 块 |
| `WanTransformer3DModel` | 类 | 完整 3D Transformer |

## 与其他模块的关系

- **`pipeline_wan2_2.py`** 等管线：调用模型进行去噪
- **`helios_transformer.py`**：Helios 是 Wan2.2 的扩展版
- **`vllm_omni.diffusion.attention.layer`**：统一注意力层
- **`vllm_omni.diffusion.distributed.sp_plan`**：序列并行计划

## 总结

`wan2_2_transformer.py` 实现了 Wan2.2 的标准 3D DiT 架构，通过 3D RoPE、AdaLN-Zero 调制和双路交叉注意力（文本+可选图像）进行视频去噪。全面使用 vLLM 张量并行层和序列并行计划，支持高效的分布式推理。
