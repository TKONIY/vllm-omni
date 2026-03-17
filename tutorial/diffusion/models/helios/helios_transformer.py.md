# `helios_transformer.py` — Helios 3D Transformer 模型实现

## 文件概述

本文件实现了 Helios 视频生成的核心 3D Transformer 模型 `HeliosTransformer3DModel`。Helios 扩展了 Wan2.2 架构，增加了多期记忆补丁（Multi-term Memory Patch）、引导交叉注意力（Guidance Cross-Attention）和历史放大（History Amplification）等机制，以支持分块长视频生成。整个模型基于 vLLM 的张量并行（TP）层实现，支持高效的分布式推理。

## 关键代码解析

### 1. Helios 风格旋转位置编码

```python
def apply_rotary_emb_helios(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)
```

`freqs_cis` 包含 `[cos_t, cos_y, cos_x, sin_t, sin_y, sin_x]` 沿最后维度拼接，支持时间和空间维度的 3D 旋转位置编码。

### 2. 分布式 RMSNorm

```python
class DistributedRMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        local_sum_sq = (x_float**2).sum(dim=-1, keepdim=True)
        if tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            tensor_model_parallel_all_reduce(global_sum_sq)
            global_count = local_count * tp_size
        rms = torch.sqrt(global_sum_sq / global_count + self.eps)
        output = (x_float / rms) * self.weight.float()
        return output.to(input_dtype)
```

在张量并行场景下，通过 all-reduce 计算全局 RMS 值，确保归一化的正确性。

### 3. 自注意力与历史放大

```python
class HeliosSelfAttention(nn.Module):
    def forward(self, hidden_states, rotary_emb, original_context_length):
        # ... QKV 投影与 RoPE 应用 ...
        if self.is_amplify_history and original_context_length is not None:
            history_seq_len = hidden_states.shape[1] - original_context_length
            if history_seq_len > 0:
                scale_key = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
                key = torch.cat(
                    [key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]],
                    dim=1,
                )
```

历史放大机制通过可学习的缩放因子 `history_key_scale` 增强历史帧的 key 信号强度，使模型在分块生成时更好地利用历史信息。

### 4. 多期记忆补丁（Multi-term Memory Patch）

```python
# 在 HeliosTransformer3DModel.__init__ 中
if has_multi_term_memory_patch:
    self.patch_short = Conv3dLayer(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    self.patch_mid = Conv3dLayer(in_channels, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
    self.patch_long = Conv3dLayer(in_channels, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
```

三种不同分辨率的历史补丁编码器：
- **短期**：`(1,2,2)` — 高分辨率，保留近期细节
- **中期**：`(2,4,4)` — 中等分辨率
- **长期**：`(4,8,8)` — 低分辨率，压缩远期信息

### 5. 引导交叉注意力

```python
class HeliosTransformerBlock(nn.Module):
    def forward(self, ...):
        if self.guidance_cross_attn and original_context_length is not None:
            history_hidden_states, current_hidden_states = (
                hidden_states[:, :history_seq_len],
                hidden_states[:, history_seq_len:],
            )
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            current_hidden_states = current_hidden_states + attn_output
            hidden_states = torch.cat([history_hidden_states, current_hidden_states], dim=1)
```

仅当前块的 token 参与文本交叉注意力，历史帧不与文本交互，避免重复注入文本语义信息。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `apply_rotary_emb_helios` | 函数 | 3D 旋转位置编码应用 |
| `DistributedRMSNorm` | 类 | TP 感知的全局 RMSNorm |
| `ColumnParallelGELU` | 类 | 列并行 + GELU 激活 |
| `HeliosFeedForward` | 类 | TP 启用的前馈网络 |
| `HeliosRotaryPosEmbed` | 类 | 3D 旋转位置编码生成器 |
| `HeliosTimeTextEmbedding` | 类 | 时间步和文本条件嵌入 |
| `HeliosOutputNorm` | 类 | 输出归一化（仅提取当前块） |
| `HeliosSelfAttention` | 类 | 自注意力（支持历史放大） |
| `HeliosCrossAttention` | 类 | 交叉注意力 |
| `HeliosTransformerBlock` | 类 | Transformer 块（自注意力+交叉注意力+FFN） |
| `HeliosTransformer3DModel` | 类 | 完整 3D Transformer 模型 |

## 与其他模块的关系

- **`vllm.model_executor.layers`**：使用 `QKVParallelLinear`、`ColumnParallelLinear`、`RowParallelLinear`、`Conv3dLayer` 等 TP 层
- **`vllm_omni.diffusion.attention.layer.Attention`**：统一注意力计算层
- **`vllm_omni.diffusion.distributed.sp_plan`**：序列并行计划
- **`pipeline_helios.py`**：被管线调用进行去噪推理

## 总结

`helios_transformer.py` 实现了 Helios 的核心 3D Transformer 架构，通过多期记忆补丁、引导交叉注意力和历史放大三大创新机制支持高质量的分块长视频生成，同时全面采用 vLLM 的张量并行层保证分布式推理效率。`load_weights` 方法处理了 QKV 融合、FFN 重映射和 TP 归一化权重的分片加载。
