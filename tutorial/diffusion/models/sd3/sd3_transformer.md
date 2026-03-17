# `sd3_transformer.py` -- Stable Diffusion 3 MMDiT Transformer

## 文件概述

实现 SD3 的 MMDiT（Multimodal Diffusion Transformer）架构。MMDiT 对文本和图像使用双路 Joint Attention，支持 Dual Attention（SD3.5 变体），并使用 PatchEmbed 将图像转为 patch 序列。

**文件路径**: `vllm_omni/diffusion/models/sd3/sd3_transformer.py`

## 关键代码解析

### SD3PatchEmbed 图像 Patch 嵌入

```python
class SD3PatchEmbed(nn.Module):
    def forward(self, latent):
        x = self.proj(latent)              # Conv2d: [B,C,H,W] -> [B,embed_dim,H',W']
        x = x.flatten(2).transpose(1, 2)   # -> [B, num_patches, embed_dim]
```

使用 2D 卷积将潜变量直接转为 patch 嵌入。

### SD3CrossAttention 联合注意力

```python
class SD3CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, added_kv_proj_dim):
        self.to_qkv = QKVParallelLinear(...)      # 图像流 QKV（融合）
        self.add_kv_proj = QKVParallelLinear(...)  # 文本流 QKV（融合）
        self.norm_q = RMSNorm(head_dim)
        self.norm_k = RMSNorm(head_dim)

    def forward(self, hidden_states, encoder_hidden_states):
        # 拼接 [text, image] 进行 Joint Attention
        query = torch.cat([txt_query, img_query], dim=1)
        key = torch.cat([txt_key, img_key], dim=1)
        value = torch.cat([txt_value, img_value], dim=1)
        hidden_states = self.attn(query, key, value)
        # 分割回文本和图像部分
```

### SD3TransformerBlock 带 Dual Attention 支持

```python
class SD3TransformerBlock(nn.Module):
    def __init__(self, ..., use_dual_attention=False):
        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)  # SD3.5 专用 AdaLN
            self.attn2 = SD3CrossAttention(added_kv_proj_dim=None, ...)  # 二次自注意力
```

SD3.5 变体在部分层中使用 Dual Attention：先做联合注意力（文本+图像），再做纯图像自注意力。

### SD3Transformer2DModel 主模型

```python
class SD3Transformer2DModel(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep):
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  # PatchEmbed + 位置编码
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb)
        # Unpatchify: [B, H'*W', C*p*p] -> [B, C, H, W]
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states.reshape(...))
```

输出时执行 unpatchify 操作，将 patch 序列还原为空间图像。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `SD3Transformer2DModel` | nn.Module | SD3 完整 Transformer |
| `SD3TransformerBlock` | nn.Module | MMDiT 块（可选 Dual Attention） |
| `SD3CrossAttention` | nn.Module | Joint Attention |
| `SD3PatchEmbed` | nn.Module | 图像 Patch 嵌入 |
| `FeedForward` | nn.Module | GELU FFN |
| `GELU` | nn.Module | 并行化 GELU |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 被使用 | `pipeline_sd3.py` | SD3 Pipeline 调用 |
| 依赖 | vLLM 并行层 | QKVParallelLinear 等 |
| 依赖 | diffusers | PatchEmbed、AdaLayerNorm 等 |

## 总结

SD3 Transformer 实现了 MMDiT 架构：文本和图像通过 Joint Attention 交互，使用 PatchEmbed 将潜变量转为序列，支持 SD3.5 的 Dual Attention 变体。与 FLUX 系列不同，SD3 没有单流块设计，所有层都是双流 Joint Attention。
