# `ring_selector.py` — Ring Attention 内核选择器

## 文件概述

`ring_selector.py` 定义了 Ring Attention 支持的注意力类型枚举 `AttnType`，以及内核选择函数 `select_flash_attn_impl`。它根据指定的注意力类型返回对应的计算内核函数，支持 11 种注意力实现。

## 关键代码解析

### 1. AttnType 枚举

```python
class AttnType(Enum):
    AITER = "aiter"              # AMD Aiter
    FA = "fa"                    # Flash Attention 2
    FA3 = "fa3"                  # Flash Attention 3
    FLASHINFER = "flashinfer"    # FlashInfer
    TORCH = "torch"             # PyTorch SDPA
    SAGE_AUTO = "sage_auto"     # SageAttention 自动选择
    SAGE_FP16 = "sage_fp16"     # SageAttention INT8 QK + FP16 PV (CUDA)
    SAGE_FP16_TRITON = "sage_fp16_triton"  # SageAttention INT8 QK + FP16 PV (Triton)
    SAGE_FP8 = "sage_fp8"       # SageAttention INT8 QK + FP8 PV
    SAGE_FP8_SM90 = "sage_fp8_sm90"  # SageAttention INT8 QK + FP8 PV (SM90 优化)
    SPARSE_SAGE = "sparse_sage"  # SparseSageAttention
```

### 2. 内核选择函数

```python
def select_flash_attn_impl(
    impl_type: AttnType,
    stage: str = "fwd-only",
    attn_processor: torch.nn.Module | None = None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
```

仅支持 `fwd-only` 阶段（推理模式，无反向传播）。根据 `impl_type` 返回对应的内核函数。

### 3. SageAttention 变体

```python
elif impl_type == AttnType.SAGE_FP16:
    return partial(
        sageattention.sageattn_qk_int8_pv_fp16_cuda,
        pv_accum_dtype="fp32",
        tensor_layout="NHD",
        return_lse=True,
    )

elif impl_type == AttnType.SAGE_FP8_SM90:
    return partial(
        sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90,
        pv_accum_dtype="fp32+fp32",
        tensor_layout="NHD",
        return_lse=True,
    )
```

SageAttention 支持多种精度组合：
- **SAGE_AUTO**：自动选择最优精度
- **SAGE_FP16 / SAGE_FP16_TRITON**：QK 用 INT8 量化，PV 用 FP16 累积
- **SAGE_FP8 / SAGE_FP8_SM90**：QK 用 INT8 量化，PV 用 FP8 累积（SM90 为 Hopper 架构优化）

所有 Sage 变体都设置 `return_lse=True`，使其与 Ring Attention 兼容。

### 4. SparseSageAttention

```python
elif impl_type == AttnType.SPARSE_SAGE:
    if not isinstance(attn_processor, SparseAttentionMeansim):
        raise ImportError(...)

    def fn(q, k, v, causal=False, softmax_scale=None, *args, **kwargs):
        return (
            attn_processor(q, k, v, is_causal=causal, scale=softmax_scale, tensor_layout="NHD"),
            None,  # LSE 为 None
        )
    return fn
```

SparseSageAttention 需要传入自定义的 `attn_processor`，且不返回 LSE（因此不支持 Ring Attention 的 LSE 累积）。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `AttnType` | 枚举 | 11 种注意力实现类型 |
| `AttnType.from_string` | 类方法 | 从字符串创建枚举值 |
| `select_flash_attn_impl` | 函数 | 根据 AttnType 返回对应的计算内核函数 |

## 与其他模块的关系

- **`ring_globals.py`**：使用 `HAS_SAGE_ATTENTION`、`HAS_SPARSE_SAGE_ATTENTION` 等标志
- **`ring_kernels.py`**：返回其中定义的内核函数
- **`ring_flash_attn.py`**：在 Ring 循环中调用 `select_flash_attn_impl` 获取内核
- **`parallel/ring.py`**：使用 `AttnType` 枚举指定 Ring Attention 的内核类型

## 总结

`ring_selector.py` 是 Ring Attention 内核选择的核心。通过 `AttnType` 枚举统一管理 11 种注意力实现，通过 `select_flash_attn_impl` 函数提供统一的选择接口。特别值得注意的是对 SageAttention 多种精度变体的支持，以及通过 `partial` 函数预配置参数的设计模式。
