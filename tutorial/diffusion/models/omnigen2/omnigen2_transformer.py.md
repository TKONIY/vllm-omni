# `omnigen2_transformer.py` — OmniGen2 Transformer 模型

## 文件概述

该文件实现了 OmniGen2 的 2D Transformer 模型，采用 Lumina2 风格的架构，包含文本 Refiner、噪声 Refiner 和主 Transformer 层。模型使用 QKV 融合投影（禁用 TP）和 vLLM-Omni 注意力层，支持旋转位置编码。

## 关键代码解析

### OmniGen2Attention

```python
class OmniGen2Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, eps=1e-5):
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim, head_size=self.head_dim,
            total_num_heads=num_heads, total_num_kv_heads=num_kv_heads,
            disable_tp=True,  # 禁用张量并行
            bias=False,
        )
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.attn = Attention(num_heads=num_heads, head_size=self.head_dim, ...)
```

注意力层使用 `disable_tp=True` 禁用张量并行，使用 QK 归一化和 RoPE。

### Transformer 块

模型包含三种 Transformer 块：

```python
# 1. 主 Transformer 块（带调制）
class OmniGen2TransformerBlock(nn.Module):
    # LuminaRMSNormZero + Attention + FeedForward

# 2. 文本 Refiner 块（无调制）
class OmniGen2TextRefinerBlock(nn.Module):
    # 无 AdaLN 调制的简单 Transformer 块

# 3. 噪声 Refiner 块（带调制）
# 与主块类似但层数不同
```

### 旋转位置编码

```python
class OmniGen2RotaryPosEmbed(nn.Module):
    def __init__(self, theta=10000, axes_dim=(32, 32, 32), axes_lens=(300, 512, 512)):
        # 预计算每个轴的频率
    def forward(self, ids):
        # 根据位置 ID 索引预计算的 cos/sin
```

### 主模型 — OmniGen2Transformer2DModel

```python
class OmniGen2Transformer2DModel(nn.Module):
    def forward(self, hidden_states, timestep, text_hidden_states, ...):
        # 1. 时间步和文本嵌入
        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states)
        # 2. Patchify 图像
        img_tokens = rearrange(hidden_states, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", ...)
        img_tokens = self.x_embedder(img_tokens)
        # 3. 分别精化
        for layer in self.context_refiner: text_hidden_states = layer(...)
        for layer in self.noise_refiner: img_tokens = layer(...)
        # 4. 联合处理
        joint_hidden_states = concat(text, img)
        for layer in self.layers: hidden_states = layer(...)
        # 5. 输出投影
        output = self.norm_out(hidden_states, temb)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniGen2Transformer2DModel` | 类 | 主 Transformer 模型 |
| `OmniGen2Attention` | 类 | 注意力层（禁用 TP） |
| `OmniGen2TransformerBlock` | 类 | 主 Transformer 块 |
| `OmniGen2RotaryPosEmbed` | 类 | 多轴 RoPE |
| `OmniGen2FeedForward` | 类 | SwiGLU 前馈网络 |

## 与其他模块的关系

- 被 `pipeline_omnigen2.py` 使用
- 使用 `vllm_omni.diffusion.attention.layer.Attention` 进行注意力计算

## 总结

OmniGen2 Transformer 采用 Lumina2 风格的分阶段处理架构：先分别精化文本和噪声表示，再联合处理。与 MammothModa2 类似但不完全相同，该模型禁用了张量并行，使用标准的线性层进行投影。
