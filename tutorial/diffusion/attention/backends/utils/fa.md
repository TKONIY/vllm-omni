# `fa.py` — Flash Attention 工具函数与导入管理

## 文件概述

`fa.py` 是 Flash Attention 生态的核心工具模块，承担两个职责：

1. **Flash Attention 导入管理**：通过回退链检测和导入可用的 Flash Attention 版本（FA3 → FA2），并适配不同平台（CUDA / ROCm / XPU）
2. **Unpad/Pad 工具**：提供变长序列的紧凑化（unpad）和恢复（pad）工具函数，用于 varlen attention

## 关键代码解析

### 1. Flash Attention 导入回退链

```python
flash_attn_func = None
flash_attn_varlen_func = None

if current_omni_platform.is_rocm():
    # ROCm: 优先尝试 Aiter
    try:
        from aiter import flash_attn_func, flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError):
        pass
elif current_omni_platform.is_xpu():
    try:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError):
        pass
else:
    # CUDA: FA3 → FA2 回退链
    try:
        from fa3_fwd_interface import flash_attn_func, flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError):
        pass

    if flash_attn_func is None:
        try:
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        except (ImportError, ModuleNotFoundError):
            pass

    if flash_attn_func is None:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except (ImportError, ModuleNotFoundError):
            pass

HAS_FLASH_ATTN = flash_attn_func is not None or flash_attn_varlen_func is not None
```

CUDA 平台的导入优先级：
1. `fa3_fwd_interface`（PyPI 的 FA3 包，支持 Ampere/Ada/Hopper）
2. `flash_attn_interface`（源码编译的 FA3）
3. `flash_attn`（FA2 标准包）
4. `flash_attn.flash_attn_interface`（FA2 子模块路径）

### 2. _unpad_input — 移除 padding

```python
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    将带 padding 的张量 (batch, seqlen, ...) 转换为紧凑张量 (total_nnz, ...)。

    返回:
        hidden_states: (total_nnz, ...) 紧凑张量
        indices: (total_nnz,) 有效 token 在展平序列中的索引
        cu_seqlens: (batch + 1,) 累积序列长度
        max_seqlen_in_batch: int 批次中最长的有效序列
        seqused: (batch,) 每个序列的有效 token 数
    """
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        _index_first_axis(hidden_states, indices),
        indices, cu_seqlens, max_seqlen_in_batch, used_seqlens_in_batch,
    )
```

### 3. _upad_input — 统一 Q/K/V unpad

```python
def _upad_input(query_layer, key_layer, value_layer, attention_mask, query_length, unpad_input_func):
    """
    统一 unpad Q/K/V，避免重复计算中间张量。

    处理三种情况：
    - query_length == kv_seq_len：Q 和 K 使用相同的索引
    - query_length == 1：单 token 查询（如自回归解码）
    - 其他：使用左 padding 假设，截取掩码的最后 query_length 部分
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

    key_layer = _index_first_axis(key_layer, indices_k)
    value_layer = _index_first_axis(value_layer, indices_k)

    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer, indices_k)
        cu_seqlens_q = cu_seqlens_k
    elif query_length == 1:
        cu_seqlens_q = torch.arange(batch_size + 1, ...)
        query_layer = query_layer.squeeze(1)
    else:
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input_func(query_layer, attention_mask)

    return (query_layer, key_layer, value_layer, indices_q,
            (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k))
```

### 4. _pad_input — 恢复 padding

```python
def _pad_input(hidden_states, indices, batch, seqlen):
    """将紧凑张量 (total_nnz, ...) 恢复为 (batch, seqlen, ...)。"""
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `flash_attn_func` | 变量 | 检测到的高层 Flash Attention 函数 |
| `flash_attn_varlen_func` | 变量 | 检测到的变长序列 Flash Attention 函数 |
| `HAS_FLASH_ATTN` | 变量 | Flash Attention 是否可用的标志 |
| `_unpad_input` | 函数 | 移除 padding，生成紧凑张量 |
| `_pad_input` | 函数 | 恢复 padding，还原标准形状 |
| `_upad_input` | 函数 | 统一的 Q/K/V unpad 函数 |
| `_get_unpad_data` | 函数 | 从掩码提取索引、累积长度等元数据 |
| `_index_first_axis` | 函数 | 在展平的第一维上进行索引 |
| `_is_packed_sequence` | 函数 | 检测是否为打包序列 |

## 与其他模块的关系

- **`flash_attn.py`**：使用 `flash_attn_func`、`flash_attn_varlen_func` 和 unpad/pad 工具
- **`utils/__init__.py`**：导出 `_pad_input`、`_unpad_input`、`_upad_input`
- **`vllm_omni.platforms`**：根据平台类型选择不同的导入路径

## 总结

`fa.py` 是 Flash Attention 生态的适配层，解决了两个核心问题：(1) 在多种 FA 版本和硬件平台间提供统一的导入接口；(2) 实现了高效的 unpad/pad 工具，支持变长序列注意力。通过回退链设计，无论用户安装的是 FA3 还是 FA2，系统都能自动选择最优版本。
