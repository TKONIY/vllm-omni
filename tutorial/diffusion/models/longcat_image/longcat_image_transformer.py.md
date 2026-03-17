# `longcat_image_transformer.py` — LongCat 图像 Transformer 模型

## 文件概述

该文件实现了 LongCat Image 的 2D Transformer 模型，基于 Flux 架构，采用双流（dual-stream）和单流（single-stream）混合 Transformer 块设计。模型支持序列并行（Sequence Parallelism），使用 vLLM 的张量并行层进行高效推理。

## 关键代码解析

### FeedForward 前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, bias: bool = True):
        inner_dim = int(dim * mult)
        self.w_in = ColumnParallelLinear(dim, inner_dim, bias=bias, return_bias=False)
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.w_out = RowParallelLinear(inner_dim, dim_out, bias=bias, return_bias=False)
```

使用 vLLM 的 `ColumnParallelLinear` / `RowParallelLinear` 实现张量并行的前馈网络。

### LongCatImageAttention 注意力层

```python
class LongCatImageAttention(nn.Module):
    def __init__(self, parallel_config, query_dim, heads, dim_head, ...):
        self.to_qkv = QKVParallelLinear(...)    # 融合 QKV 投影
        self.add_kv_proj = QKVParallelLinear(...)  # 交叉注意力的额外 KV 投影
        self.attn = Attention(...)               # vLLM-Omni 注意力层
```

核心注意力实现，支持两种模式：
- **双流模式**（added_kv_proj_dim 不为 None）：图像和文本各自有 QKV 投影，通过 joint attention 联合处理
- **单流模式**：文本和图像拼接后统一进行注意力计算

### 序列并行 RoPE 处理

```python
def _sp_attention_with_rope(self, img_query, img_key, img_value,
                            text_query, text_key, text_value, ...):
    # 分别对文本和图像应用 RoPE
    img_query = apply_rotary_emb(img_query, img_rotary_emb_split, sequence_dim=1)
    text_query = apply_rotary_emb(text_query, txt_rotary_emb, sequence_dim=1)
    # 使用 joint_strategy="front" 进行联合注意力
    return self.attn(img_query, img_key, img_value,
                     AttentionMetadata(joint_query=text_query, ...))
```

在 SP 模式下，文本部分保持完整（复制到所有 rank），图像部分被分片（chunked）。

### RoPEPreparer 模块

```python
class RoPEPreparer(nn.Module):
    def forward(self, txt_ids, img_ids):
        # 返回 (txt_cos, txt_sin, img_cos, img_sin)
        # 输出顺序与 _sp_plan 索引对应：
        # 0,1 是文本（复制），2,3 是图像（分片）
```

将 RoPE 计算封装到独立模块中，使 `_sp_plan` 可以自动对图像部分的旋转位置编码进行分片。

### _sp_plan 序列并行计划

```python
_sp_plan = {
    "": {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3)},
    "rope_preparer": {
        2: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
        3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
    },
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

定义序列并行的输入分片和输出聚合策略。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FeedForward` | 类 | 张量并行的前馈网络 |
| `LongCatImageAttention` | 类 | 支持 SP 的联合注意力层 |
| `LongCatImageTransformerBlock` | 类 | 双流 Transformer 块 |
| `LongCatImageSingleTransformerBlock` | 类 | 单流 Transformer 块 |
| `LongCatImagePosEmbed` | 类 | 多轴旋转位置编码 |
| `RoPEPreparer` | 类 | SP 感知的 RoPE 准备模块 |
| `LongCatImageTimestepEmbeddings` | 类 | 时间步嵌入 |
| `LongCatImageTransformer2DModel` | 类 | 主 Transformer 模型 |

## 与其他模块的关系

- 使用 `vllm_omni.diffusion.attention.layer.Attention` 进行高效注意力计算
- 使用 `vllm_omni.diffusion.distributed.sp_plan` 中的 SP 原语
- 被 `pipeline_longcat_image.py` 和 `pipeline_longcat_image_edit.py` 引用
- `load_weights` 方法处理 diffusers 到 vLLM 的权重映射

## 总结

该文件实现了完整的 LongCat Image Transformer 模型，采用 Flux 架构的双流/单流混合设计。其最大特点是深度集成了 vLLM 的张量并行和序列并行功能，通过 `_sp_plan` 声明式地定义并行策略，使模型可以高效地在多 GPU 上进行推理。
