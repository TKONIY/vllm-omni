# `z_image_transformer.py` — Z-Image Transformer 模型

## 文件概述

该文件实现了阿里巴巴 Z-Image 的 Transformer 模型，是本项目中并行化最为完整的模型之一。支持完整的张量并行（TP）和序列并行（SP），使用 vLLM 的量化层，采用噪声/文本分别精化 + 联合处理的架构。

## 关键代码解析

### TP 约束验证

```python
def validate_zimage_tp_constraints(*, dim, n_heads, n_kv_heads, ...):
    # 验证所有并行维度的整除性
    if dim % tp_size != 0: raise ValueError(...)
    if n_heads % tp_size != 0: raise ValueError(...)
    # 返回支持的 TP 候选值
    supported_tp_candidates = sorted(
        _positive_divisors(n_heads) & _positive_divisors(n_kv_heads) & ...
    )
```

严格验证 TP 约束，并返回所有支持的 TP 大小，便于用户选择。

### ZImageAttention — 完全 TP 感知

```python
class ZImageAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, ...):
        self.to_qkv = QKVParallelLinear(...)     # 列并行 QKV
        self.to_out = nn.ModuleList([
            RowParallelLinear(dim, dim, input_is_parallel=True, ...)  # 行并行输出
        ])
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,      # 使用 TP 后的 head 数
            num_kv_heads=self.to_qkv.num_kv_heads,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)
```

完整的 TP 链路：QKV 列并行分片 -> 注意力 -> 行并行输出（自动 all-reduce）。

### FeedForward — TP 感知 SwiGLU

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.w13 = MergedColumnParallelLinear(dim, [hidden_dim]*2, ...)  # gate + up
        self.act = SiluAndMul()                                          # SwiGLU
        self.w2 = RowParallelLinear(hidden_dim, dim, input_is_parallel=True, ...)
```

### UnifiedPrepare — SP 支持模块

```python
class UnifiedPrepare(nn.Module):
    def forward(self, x, x_cos, x_sin, cap_feats, cap_cos, cap_sin, ...):
        # 将图像和文本序列拼接为统一序列
        for i in range(bsz):
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
        # 填充到相同长度
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        # 创建注意力掩码
        unified_attn_mask[i, :seq_len] = 1
        return unified, unified_cos, unified_sin, unified_attn_mask
```

封装统一序列准备逻辑到独立模块，使 `_sp_plan` 可以自动分片输出。

### _sp_plan 序列并行计划

```python
_sp_plan = {
    "unified_prepare": {
        0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified
        1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # cos
        2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # sin
        3: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True),  # mask
    },
    "all_final_layer.2-1": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

### Patchify 和嵌入

```python
def patchify_and_embed(self, all_image, all_cap_feats, patch_size, f_patch_size):
    # 处理图像：patchify + padding + 位置 ID
    image = image.permute(1,3,5,2,4,6,0).reshape(F*H*W, pF*pH*pW*C)
    # 处理文本：padding + 位置 ID
    cap_padded_pos_ids = self.create_coordinate_grid(size=(cap_len+pad, 1, 1), ...)
```

### 量化支持

```python
class ZImageTransformer2DModel(CachedTransformer):
    def __init__(self, ..., quant_config=None):
        # 所有线性层接受 quant_config 参数
        self.t_embedder = TimestepEmbedder(..., quant_config=quant_config)
        self.cap_embedder = nn.Sequential(
            RMSNorm(...),
            ReplicatedLinear(..., quant_config=quant_config),
        )
```

### 前向传播

```python
def forward(self, x: list[torch.Tensor], t, cap_feats: list[torch.Tensor], ...):
    # 1. Patchify + 嵌入
    x, cap_feats, ... = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
    # 2. 噪声精化
    for layer in self.noise_refiner: x = layer(x, x_attn_mask, x_cos, x_sin, adaln_input)
    # 3. 文本精化
    for layer in self.context_refiner: cap_feats = layer(cap_feats, cap_attn_mask, cap_cos, cap_sin)
    # 4. 统一准备（SP 分片点）
    unified, unified_cos, unified_sin, unified_attn_mask = self.unified_prepare(...)
    # 5. 主 Transformer
    for layer in self.layers: unified = layer(unified, unified_attn_mask, unified_cos, unified_sin, adaln_input)
    # 6. 最终层 + unpatchify
    unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
    x = self.unpatchify(unified, x_size, patch_size, f_patch_size)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ZImageTransformer2DModel` | 类 | 主 Transformer 模型 |
| `ZImageAttention` | 类 | 完全 TP 感知的注意力层 |
| `ZImageTransformerBlock` | 类 | Transformer 块（AdaLN 调制） |
| `FeedForward` | 类 | TP 感知的 SwiGLU FFN |
| `FinalLayer` | 类 | 最终投影层 |
| `TimestepEmbedder` | 类 | 时间步嵌入器 |
| `RopeEmbedder` | 类 | 多轴 RoPE 嵌入器（带缓存） |
| `UnifiedPrepare` | 类 | SP 感知的统一序列准备模块 |
| `validate_zimage_tp_constraints` | 函数 | TP 约束验证 |

## 与其他模块的关系

- 被 `pipeline_z_image.py` 使用
- 继承 `CachedTransformer` 支持推理缓存
- 使用 `_sp_plan` 实现序列并行
- 使用 vLLM 的量化配置

## 总结

Z-Image Transformer 是本项目中并行化最完整的模型，同时支持 TP、SP 和量化。其设计亮点包括：(1) 严格的 TP 约束验证；(2) 通过 UnifiedPrepare 模块实现 SP 的分片/聚合；(3) 多 patch size 支持（通过 ModuleDict 管理不同的 embedder 和 final_layer）；(4) 带缓存的 RoPE 嵌入器避免重复计算。
