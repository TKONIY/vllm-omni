# `rope_real.py` — 实值旋转位置编码

## 文件概述

该文件实现了 MammothModa2 使用的实值（real-valued）旋转位置编码（RoPE），与传统的复数 RoPE 不同，这里使用实数对表示旋转。包含 RoPE 的应用函数和多轴位置编码的计算模块。

## 关键代码解析

### 实值 RoPE 应用

```python
def apply_real_rotary_emb(x, freqs_cos, freqs_sin):
    # 将输入分为成对的元素
    x_reshaped = x.view(batch, seq_len, num_heads, dim // 2, 2)
    # cos/sin 也分为两组
    cos_1 = freqs_cos_reshaped[..., 0]
    sin_1 = freqs_sin_reshaped[..., 0]
    # 应用旋转（2x2 旋转矩阵）
    out1 = x1 * cos_1 - x2 * sin_1
    out2 = x1 * sin_2 + x2 * cos_2
    out = torch.stack([out1, out2], dim=-1)
```

使用实数对 `(cos, sin)` 的两个分量分别作用于输入的奇偶维度对，实现旋转变换。

### 预计算频率表

```python
@staticmethod
def get_freqs_real(axes_dim, axes_lens, theta):
    freqs_real = []
    for d, e in zip(axes_dim, axes_lens):
        cos_emb, sin_emb = get_1d_rotary_pos_embed_real(d, e, theta=theta)
        freqs_real.append((cos_emb, sin_emb))
    return freqs_real
```

为每个轴（文本、高度、宽度）独立预计算 RoPE 频率。

### 多模态位置 ID 构建

```python
class RotaryPosEmbedReal(nn.Module):
    def forward(self, freqs_real, attention_mask, l_effective_ref_img_len, ...):
        # 构建位置 ID：3D 坐标 (text_pos, row, col)
        position_ids[i, :cap_seq_len] = repeat(torch.arange(cap_seq_len), "l -> l 3")
        # 图像使用 2D 网格坐标
        row_ids = repeat(torch.arange(H_tokens), "h -> h w", w=W_tokens).flatten()
        col_ids = repeat(torch.arange(W_tokens), "w -> h w", h=H_tokens).flatten()
        # 分别输出：文本 RoPE、参考图像 RoPE、噪声图像 RoPE、联合 RoPE
        return (cap_freqs, ref_img_freqs, img_freqs, joint_freqs, ...)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `apply_real_rotary_emb` | 函数 | 应用实值 RoPE 到张量 |
| `get_1d_rotary_pos_embed_real` | 函数 | 生成 1D 实值 RoPE |
| `RotaryPosEmbedReal` | 类 | 多轴 RoPE 计算模块 |

## 与其他模块的关系

- 被 `mammothmoda2_dit_model.py` 中的 `AttnProcessor` 和 `Transformer2DModel` 使用
- 被 `pipeline_mammothmoda2_dit.py` 通过 `gen_freqs_cis` 预计算频率

## 总结

该文件实现了 MammothModa2 的实值 RoPE 系统，支持文本（1D）、图像（2D 网格）和参考图像的多模态位置编码。通过预计算频率表和索引查找，高效地为不同模态的序列生成旋转位置编码。
