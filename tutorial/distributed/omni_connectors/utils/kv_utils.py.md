# `kv_utils.py` — KV 缓存工具函数

## 文件概述

该文件提供 KV 缓存操作的工具函数，核心功能是统一不同注意力后端返回的 KV 缓存布局。

## 关键代码解析

### normalize_layer_kv — KV 布局标准化

```python
def normalize_layer_kv(layer_kv, *, req_id="", layer_idx=-1):
    """将单层 KV 缓存标准化为 (key_blocks, value_blocks) 元组"""
```

vLLM 不同注意力后端返回的 KV 缓存布局不同：

| 后端 | 形状 | KV 维度位置 |
|------|------|-----------|
| FlashAttention | `(2, num_blocks, block_size, num_kv_heads, head_size)` | dim 0 |
| FlashInfer | `(num_blocks, 2, block_size, num_kv_heads, head_size)` | dim 1 |
| Tuple | `(key_tensor, value_tensor)` | N/A |

```python
if isinstance(layer_kv, torch.Tensor):
    if layer_kv.ndim >= 3 and layer_kv.shape[0] == 2:
        # FlashAttention 布局：dim-0 选择 key/value
        key_blocks = layer_kv[0]
        value_blocks = layer_kv[1]
    elif layer_kv.ndim >= 3 and layer_kv.shape[1] == 2:
        # FlashInfer 布局：dim-1 选择 key/value
        key_blocks = layer_kv[:, 0]
        value_blocks = layer_kv[:, 1]
elif isinstance(layer_kv, tuple):
    if len(layer_kv) == 2:
        key_blocks, value_blocks = layer_kv
```

还会验证：
- Tensor 类型和维度正确性
- 至少 2D（用于 block 索引）

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `normalize_layer_kv()` | function | 统一 KV 缓存布局为 `(key, value)` 元组 |
| `LayerKV` | type alias | `torch.Tensor | tuple[torch.Tensor, torch.Tensor]` |

## 与其他模块的关系

- 被 `kv_transfer_manager.py` 中的 `_extract_kv_cache()` 调用
- 确保下游代码可以处理任何注意力后端的 KV 输出

## 总结

`normalize_layer_kv` 是一个关键的兼容性工具，屏蔽了不同注意力后端（FlashAttention、FlashInfer 等）的 KV 缓存布局差异，为 KV 缓存提取和传输提供统一的数据格式。
