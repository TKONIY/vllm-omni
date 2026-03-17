# `flux2_klein_transformer.py` -- Flux 2 Klein Transformer（序列并行版）

## 文件概述

实现支持序列并行（Sequence Parallel, SP）的 Flux 2 Transformer。核心架构与 `flux2_transformer.py` 一致，但所有注意力和投影层增加了 SP 相关逻辑：图像 token 按 SP rank 分片，文本 token 在所有 rank 间复制。通过 `_sp_plan` 声明式配置自动处理序列分片和聚合。

**文件路径**: `vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py`

## 关键代码解析

### _sp_plan 序列并行配置

```python
class Flux2Transformer2DModel(nn.Module):
    _sp_plan = {
        "": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "rope_prepare": {
            2: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
            3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

SP 计划定义了：
- **输入层**: `hidden_states` 沿序列维度（dim=1）自动分片
- **RoPE**: 图像的 cos/sin（输出索引 2, 3）按序列分片，文本的（索引 0, 1）复制
- **输出层**: `proj_out` 聚合所有 rank 的输出

### Flux2RopePrepare RoPE 预计算模块

```python
class Flux2RopePrepare(nn.Module):
    def forward(self, img_ids, txt_ids):
        img_freqs_cos, img_freqs_sin = self.pos_embed(img_ids)
        txt_freqs_cos, txt_freqs_sin = self.pos_embed(txt_ids)
        return txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin
```

将 RoPE 计算封装为独立模块，便于 SP 计划对文本和图像的 RoPE 进行不同的分片/复制策略。

### SP 感知的联合注意力

```python
class Flux2Attention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, ...):
        sp_size = self.parallel_config.sequence_parallel_size
        use_sp_joint_attention = sp_size > 1 and not forward_ctx.split_text_embed_in_sp
        if use_sp_joint_attention:
            # 分别对文本和图像 RoPE，使用 joint_strategy="front"
            attn_metadata = AttentionMetadata(
                joint_query=encoder_query, joint_key=encoder_key,
                joint_value=encoder_value, joint_strategy="front")
            hidden_states = self.attn(query, key, value, attn_metadata)
```

在 SP 模式下，文本 KV 不分片而是作为 `joint` 部分在所有 rank 上复制参与注意力计算。

### SP padding 处理

```python
ctx = get_forward_context()
if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
    hidden_states_mask = torch.ones(batch_size, img_padded_seq_len, ...)
    hidden_states_mask[:, ctx.sp_original_seq_len:] = False
```

当序列长度不能被 SP size 整除时，自动填充 padding 并通过 mask 排除。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2Transformer2DModel` | nn.Module | 支持 SP 的 Flux 2 Transformer |
| `Flux2TransformerBlock` | nn.Module | SP 感知的双流块 |
| `Flux2SingleTransformerBlock` | nn.Module | SP 感知的单流块 |
| `Flux2Attention` | nn.Module | SP 感知的联合注意力 |
| `Flux2ParallelSelfAttention` | nn.Module | SP 感知的并行自注意力 |
| `Flux2RopePrepare` | nn.Module | RoPE 预计算（SP 分片友好） |
| `Flux2Modulation` | nn.Module | 全局调制参数 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_flux2_klein.py` | Klein Pipeline 调用 |
| 依赖 | `vllm_omni.diffusion.distributed.sp_plan` | 序列并行计划 |
| 依赖 | `vllm_omni.diffusion.forward_context` | SP 上下文管理 |
| 对比 | `flux2/flux2_transformer.py` | 非 SP 版本 |

## 总结

Flux 2 Klein Transformer 是 Flux 2 的序列并行增强版。通过声明式 `_sp_plan` 和 `forward_context` 机制，实现了图像 token 沿序列维度的自动分片和聚合，文本 token 跨 rank 复制。这使得高分辨率图像（大量图像 token）可以分布在多个 GPU 上并行计算注意力，显著降低单卡显存占用。
