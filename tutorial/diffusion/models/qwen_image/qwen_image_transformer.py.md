# `qwen_image_transformer.py` — QwenImage 2D Transformer 模型

## 文件概述

本文件实现了 QwenImage 的核心 2D Transformer 去噪模型 `QwenImageTransformer2DModel`。模型采用 DiT（Diffusion Transformer）架构，使用 3D RoPE 位置编码、AdaLN-Zero 调制和交叉注意力进行文本条件化的图像去噪。支持张量并行和缓存优化。

## 关键代码解析

### 1. 位置编码准备

```python
class ImageRopePrepare(nn.Module):
    # 为图像 token 准备 2D/3D RoPE 位置编码
    # 根据 img_shapes 计算每个图像的空间位置

class ModulateIndexPrepare(nn.Module):
    # 准备 AdaLN 调制索引
    # 将时间步嵌入广播到所有 token
```

### 2. 嵌入层

```python
class QwenEmbedLayer3DRope(nn.Module):
    # 3D RoPE 嵌入层：处理时间+高度+宽度三个维度
    # 使用 Conv3dLayer 进行 patch embedding

class QwenEmbedRope(nn.Module):
    # 标准 RoPE 嵌入层
    # 使用 ColumnParallelLinear 进行 patch embedding
```

### 3. 前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim):
        self.net_0 = ColumnParallelApproxGELU(dim, inner_dim)
        self.net_2 = RowParallelLinear(inner_dim, dim, ...)
    # 使用张量并行的列/行并行线性层
```

### 4. 交叉注意力

```python
class QwenImageCrossAttention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, ...):
        # 1. 输入层归一化（AdaLN-Zero 调制）
        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa

        # 2. 自注意力 + RoPE
        attn_output = self.attn1(norm_hidden_states, rotary_emb)

        # 3. 交叉注意力（文本条件）
        attn_output = self.attn2(hidden_states, encoder_hidden_states)

        # 4. 前馈网络
        ff_output = self.ff(norm_hidden_states)
```

每个注意力块包含自注意力（带 RoPE）+ 文本交叉注意力 + FFN，通过 AdaLN-Zero 实现时间步条件化。

### 5. Transformer 模型

```python
class QwenImageTransformer2DModel(CachedTransformer):
    def __init__(self, ...):
        self.embed_layer = QwenEmbedLayer3DRope(...)  # 或 QwenEmbedRope
        self.blocks = nn.ModuleList([QwenImageTransformerBlock(...) for _ in range(num_layers)])
        self.proj_out = nn.Linear(inner_dim, out_channels * prod(patch_size))

    def forward(self, hidden_states, timestep, encoder_hidden_states, ...):
        # 1. Patch embedding + 位置编码
        # 2. 时间步嵌入
        # 3. Transformer blocks
        # 4. 输出投影 + unpatchify
```

继承自 `CachedTransformer` 以支持注意力缓存优化。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ImageRopePrepare` | 类 | 图像 RoPE 准备 |
| `ModulateIndexPrepare` | 类 | 调制索引准备 |
| `QwenTimestepProjEmbeddings` | 类 | 时间步投影嵌入 |
| `QwenEmbedLayer3DRope` | 类 | 3D RoPE 嵌入层 |
| `QwenEmbedRope` | 类 | 标准 RoPE 嵌入层 |
| `FeedForward` | 类 | TP 并行前馈网络 |
| `QwenImageCrossAttention` | 类 | 交叉注意力块 |
| `QwenImageTransformerBlock` | 类 | Transformer 块 |
| `QwenImageTransformer2DModel` | 类 | 完整 2D Transformer |

## 与其他模块的关系

- **`pipeline_qwen_image.py`** 等管线：调用 Transformer 进行去噪
- **`vllm_omni.diffusion.attention.layer.Attention`**：统一注意力计算
- **`vllm_omni.diffusion.cached_transformer`**：继承 `CachedTransformer` 基类
- **vLLM TP 层**：张量并行支持

## 总结

`qwen_image_transformer.py` 实现了基于 DiT 架构的 2D Transformer 去噪模型，通过 3D RoPE 保留时空结构、AdaLN-Zero 实现高效的时间步条件化、交叉注意力注入文本语义，全面使用 vLLM 张量并行层支持分布式推理。
