# `wan2_2.py` — DreamID-Omni 定制 Wan 模型

## 文件概述

本文件实现了 DreamID-Omni 定制版的 Wan Transformer 模型 `WanModel` 及其注意力组件。与标准 `wan2_2/wan2_2_transformer.py` 不同，本文件使用 `dreamid_omni` 外部包的基础模块（如 `WanLayerNorm`、`WanRMSNorm`、`rope_apply`、`sinusoidal_embedding_1d` 等），并添加了文本到音频（T2A）和图像到视频（TI2V）两种交叉注意力变体。

## 关键代码解析

### 1. 自注意力

```python
class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(num_heads=self.num_heads, head_size=self.head_dim, ...)
```

使用标准线性层（非 TP 并行版本）进行 QKV 投影，通过统一 `Attention` 层计算注意力。

### 2. T2V 交叉注意力

```python
class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return q, k, v
```

继承自自注意力，Q 来自源序列，K/V 来自上下文（文本特征）。

### 3. I2V 交叉注意力

```python
class WanI2VCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, ...):
        # 继承 T2V 的基础 QKV
        # 额外添加图像 KV 投影
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps)
```

在文本 KV 投影的基础上额外添加图像 KV 投影，支持双条件交叉注意力。

### 4. 注意力块

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, model_type, ...):
        self.self_attn = WanSelfAttention(dim, num_heads, ...)
        if model_type in ("t2v", "t2a"):
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, ...)
        elif model_type == "ti2v":
            self.cross_attn = WanI2VCrossAttention(dim, num_heads, ...)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
```

### 5. WanModel

```python
class WanModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, model_type="t2v", in_dim=48, dim=3072, ...):
        self.patch_embedding = nn.Conv1d(...) if len(patch_size) == 1 else nn.Conv3d(...)
        self.condition_embedder = ...  # 时间步+文本嵌入
        self.blocks = nn.ModuleList([WanAttentionBlock(...) for _ in range(num_layers)])

    def prepare_transformer_block_kwargs(self, x, t, context, seq_len, ...):
        # 准备 Transformer 块的输入参数
        # 包含 patch embedding、RoPE 频率计算等

    def post_transformer_block_out(self, x, grid_sizes, e):
        # 输出归一化 + 反 patch
```

提供 `prepare_transformer_block_kwargs` 和 `post_transformer_block_out` 接口，供 `FusionModel` 在块级别精细控制前向传播。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `WanSelfAttention` | 类 | 自注意力（QK Norm + RoPE） |
| `WanT2VCrossAttention` | 类 | 文本到视频/音频交叉注意力 |
| `WanI2VCrossAttention` | 类 | 图像到视频交叉注意力（双条件） |
| `WanAttentionBlock` | 类 | 注意力块（自注意力+交叉注意力+FFN+AdaLN） |
| `WanModel` | 类 | 完整 Wan Transformer（视频或音频） |

## 与其他模块的关系

- **`fusion.py`**：`FusionModel` 使用两个 `WanModel` 实例（视频+音频）
- **`dreamid_omni` 外部包**：基础模块（RoPE、LayerNorm 等）
- **与 `wan2_2/wan2_2_transformer.py` 的区别**：本文件不使用 vLLM TP 并行层，适配 DreamID-Omni 的融合架构

## 总结

`wan2_2.py` 实现了 DreamID-Omni 定制版的 Wan Transformer，区分于标准 Wan2.2 实现的核心在于：使用 `dreamid_omni` 外部包的基础组件、支持 T2V/T2A/TI2V 三种交叉注意力模式、提供块级别的接口供融合模型精细控制。
