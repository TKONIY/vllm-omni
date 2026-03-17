# `siglip2.py` — Siglip2 视觉编码器

## 文件概述

实现 Siglip2 视觉 Transformer 编码器和 LightProjector 投影器。Siglip2 使用动态分辨率位置嵌入，支持不同尺寸的输入图像。

## 关键代码解析

### 动态位置嵌入

```python
class Siglip2VisionEmbeddings(nn.Module):
    @staticmethod
    def resize_positional_embeddings(positional_embeddings, spatial_shapes, max_length):
        """将位置嵌入双线性插值到图像实际尺寸"""
        for i in range(batch_size):
            height, width = spatial_shapes[i]
            resized = F.interpolate(positional_embeddings,
                                     size=(height, width), mode="bilinear")
            resulted[i, :height*width] = resized.reshape(embed_dim, -1).transpose(0, 1)
```

### 注意力实现

```python
class Siglip2SdpaAttention(Siglip2Attention):
    """SDPA 优化的注意力，支持 Flash Attention 和 Efficient kernel"""
    def forward(self, hidden_states, attention_mask=None):
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask, is_causal=False)
```

### 多头注意力池化

```python
class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """使用可学习 probe 的注意力池化"""
    def forward(self, hidden_state, attention_mask=None):
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)
        hidden_state = residual + self.mlp(self.layernorm(hidden_state))
        return hidden_state[:, 0]
```

### LightProjector

```python
class LightProjector(nn.Module):
    """轻量级 MLP 投影器（Linear 或 MLP-GELU）"""
    def __init__(self, config):
        if config.projector_type == "mlp_gelu":
            modules = [nn.Linear(input_dim, n_embed)]
            for _ in range(1, depth):
                modules += [nn.GELU(), nn.Linear(n_embed, n_embed)]
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `Config` | 类 | 简单的字典到属性的配置封装 |
| `Siglip2VisionEmbeddings` | 类 | 视觉嵌入（patch + 位置） |
| `Siglip2Attention` | 类 | 标准多头注意力 |
| `Siglip2SdpaAttention` | 类 | SDPA 优化的注意力 |
| `Siglip2MLP` | 类 | FFN 层 |
| `Siglip2EncoderLayer` | 类 | 编码器层（LN-Attn-LN-MLP） |
| `Siglip2Encoder` | 类 | 完整编码器 |
| `Siglip2MultiheadAttentionPoolingHead` | 类 | 注意力池化头 |
| `Siglip2VisionTransformer` | 类 | 完整视觉 Transformer |
| `LightProjector` | 类 | 特征投影器 |

## 总结

Siglip2 是一个支持动态分辨率的视觉编码器，核心创新在于位置嵌入的双线性插值策略。LightProjector 提供灵活的特征投影能力，支持单层 Linear 和多层 MLP-GELU 两种模式。
