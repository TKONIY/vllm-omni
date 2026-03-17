# `ring_kernels.py` — Ring Attention 计算内核集合

## 文件概述

`ring_kernels.py` 集中实现了 Ring Attention 可用的各种注意力计算内核。每个内核函数都返回 `(output, log_sum_exp)` 元组，以便 Ring Attention 的 `update_out_and_lse` 正确合并分块结果。支持的内核包括：PyTorch SDPA、Flash Attention 2、Flash Attention 3、FlashInfer 和 AMD Aiter。

## 关键代码解析

### 1. PyTorch SDPA 内核

```python
def pytorch_attn_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,
                         op_type="efficient"):
    # float32 自动降级
    if op_type == "flash" and q.dtype == torch.float32:
        op_type = "efficient"

    q = q.transpose(1, 2)  # (B, S, H, D) -> (B, H, S, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if op_type == "flash":
        out, lse = _scaled_dot_product_flash_attention(q, k, v, ...)[:2]
    elif op_type == "efficient":
        out, lse = _scaled_dot_product_efficient_attention(q, k, v, ...)[:2]

    out = out.transpose(1, 2)
    return out, lse
```

支持两种底层 PyTorch 算子：
- `flash`：使用 `torch.ops.aten._scaled_dot_product_flash_attention`
- `efficient`：使用 `torch.ops.aten._scaled_dot_product_efficient_attention`（需要 `compute_log_sumexp=True` 以返回 LSE）

### 2. Flash Attention 2 内核

```python
def flash_attn_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, ...):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if flash_attn.__version__ < "2.6.3":
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, ...)
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
            window_size_left=window_size[0], window_size_right=window_size[1], ...)
    return block_out, block_lse
```

关键点：
- 使用 FA2 的底层 API `_flash_attn_forward`（而非高层 `flash_attn_func`），因为需要获取 `softmax_lse`
- 兼容 FA2 不同版本的 API 差异（`< 2.6.3` 返回 8 个值，`>= 2.6.3` 返回 4 个值且窗口参数名变化）

### 3. Flash Attention 3 内核

```python
def fa3_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, ...):
    assert HAS_FA3, "FA3 is not available"
    out, softmax_lse, *_ = fa3_fwd_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0] if window_size else -1,
        window_size_right=window_size[1] if window_size else -1,
        softcap=softcap if softcap else 0.0,
    )
    return out, softmax_lse
```

FA3 同样使用底层 API，始终返回 `softmax_lse`。

### 4. FlashInfer 内核

```python
def flashinfer_attn_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, ...):
    assert HAS_FLASHINFER, "FlashInfer is not available"
    out, lse = single_prefill_with_kv_cache(
        q[0], k[0], v[0],
        sm_scale=softmax_scale,
        causal=causal,
        return_lse=True,
    )
    lse = lse.transpose(0, 1)
    lse = lse / _LOG2_E  # FlashInfer 返回 log2 scale 的 LSE，需转换
    return out, lse
```

FlashInfer 的特殊处理：
- 仅支持 batch_size=1
- LSE 以 log2 为底返回，需要除以 `log2(e)` 转换为自然对数

### 5. AMD Aiter 内核

```python
def flash_attn_forward_aiter(q, k, v, ...):
    assert HAS_AITER, "Aiter is not available"
    block_out, block_lse = flash_attn_func_aiter(
        q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, return_lse=True,
    )
    return block_out, block_lse
```

AMD 平台专用内核，通过 `return_lse=True` 获取 LSE。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `pytorch_attn_forward` | 函数 | PyTorch SDPA 内核，支持 flash/efficient 两种模式 |
| `flash_attn_forward` | 函数 | Flash Attention 2 底层内核 |
| `fa3_forward` | 函数 | Flash Attention 3 底层内核 |
| `flash_attn3_func_forward` | 别名 | `fa3_forward` 的遗留别名 |
| `flashinfer_attn_forward` | 函数 | FlashInfer 内核（需 log2→ln 转换） |
| `flash_attn_forward_aiter` | 函数 | AMD Aiter 内核 |

## 与其他模块的关系

- **`ring_globals.py`**：导入可用性标志和底层函数
- **`ring_selector.py`**：通过 `select_flash_attn_impl` 选择这些内核
- **`ring_flash_attn.py`**：在 Ring 循环中调用被选中的内核
- **`ring_pytorch_attn.py`**：直接使用 `pytorch_attn_forward`

## 总结

`ring_kernels.py` 是 Ring Attention 的计算内核库，统一了 5 种不同注意力实现的调用接口。每个内核都返回 `(output, lse)` 元组，这是 Ring Attention 正确合并分块结果的关键。文件还处理了各库的版本差异（如 FA2 < 2.6.3 的 API 变化）和特殊行为（如 FlashInfer 的 log2 scale LSE）。
