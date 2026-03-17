# `rope.py` — 旋转位置编码

## 文件概述

`rope.py` 实现了旋转位置编码（Rotary Position Embedding, RoPE），这是 Transformer 模型中广泛使用的位置编码方法。它继承 `CustomOp` 基类，支持 CUDA（vllm_flash_attn）、ROCm（flash_attn triton）、NPU（mindiesd）等多个平台的高性能实现，并提供 PyTorch 原生回退。

## 关键代码解析

### rotate_half — 旋转辅助函数

```python
def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)
```

支持两种旋转模式：
- **GPT-NeoX 风格** (`interleaved=False`)：前半和后半交换
- **GPT-J 风格** (`interleaved=True`)：偶数和奇数位置交替

### apply_rotary_emb_torch — PyTorch 原生实现

```python
def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """x: (batch_size, seqlen, nheads, headdim)"""
    ro_dim = cos.shape[-1] * 2
    cos = repeat(cos, "... d -> ... 1 (2 d)")
    sin = repeat(sin, "... d -> ... 1 (2 d)")
    return torch.cat([
        x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
        x[..., ro_dim:],
    ], dim=-1)
```

标准 RoPE 公式：`x_rotated = x * cos + rotate_half(x) * sin`，仅对 `ro_dim` 维度的前部分应用旋转。

### RotaryEmbedding — 多平台 RoPE

```python
class RotaryEmbedding(CustomOp):
    def __init__(self, is_neox_style=False):
        super().__init__()
        self.interleaved = not is_neox_style
        # 尝试加载 flash_attn 的 triton 实现
        if find_spec("flash_attn") is not None:
            from flash_attn.ops.triton.rotary import apply_rotary
            self.apply_rotary_emb_flash_attn = apply_rotary

    def forward_cuda(self, x, cos, sin):
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
        return apply_rotary_emb(x, cos, sin, interleaved=self.interleaved)

    def forward_npu(self, x, cos, sin):
        if find_spec("mindiesd"):
            return apply_rotary_emb_mindiesd(x, cos, sin, self.interleaved)
        return self.forward_native(x, cos, sin)
```

各平台使用的实现：
- **CUDA**：vllm_flash_attn 的 CUDA kernel
- **ROCm**：优先使用 flash_attn 的 triton 实现，回退到 CUDA
- **NPU**：mindiesd 的融合算子
- **其他**：PyTorch 原生实现

### apply_rope_to_qk — 便捷应用函数

```python
def apply_rope_to_qk(rope, query, key, image_rotary_emb):
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        query = rope(query, cos.to(query.dtype), sin.to(query.dtype))
        key = rope(key, cos.to(query.dtype), sin.to(query.dtype))
    return query, key
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `RotaryEmbedding` | 类 | 多平台 RoPE 实现，继承 CustomOp |
| `rotate_half` | 函数 | 旋转辅助函数，支持 NeoX 和 GPT-J 风格 |
| `apply_rotary_emb_torch` | 函数 | PyTorch 原生 RoPE 实现 |
| `apply_rotary_emb_mindiesd` | 函数 | NPU（MindIE）RoPE 实现 |
| `apply_rope_to_qk` | 函数 | 便捷函数，对 Q/K 张量应用 RoPE |

## 与其他模块的关系

- 继承 `layers/custom_op.py` 的 `CustomOp` 基类
- 被扩散 Transformer 模型的注意力层使用
- CUDA 实现依赖 vLLM 的 `vllm_flash_attn` 库

## 总结

`rope.py` 提供了 RoPE 的完整多平台实现。通过 `CustomOp` 的调度机制，在不同硬件上自动选择最优的实现：CUDA 上使用 vllm_flash_attn kernel，ROCm 上使用 triton kernel，NPU 上使用 mindiesd 融合算子。支持 GPT-NeoX 和 GPT-J 两种旋转风格。
