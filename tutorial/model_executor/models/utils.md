# `utils.py` -- 模型通用工具函数

## 文件概述

`utils.py` 提供了一组模型级别的通用工具函数，包括权重名称前缀处理、张量分组以及安全的 reshape 操作。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/models/utils.py`

## 关键代码解析

### add_prefix_to_loaded_weights

为已加载权重的名称添加前缀，常用于多阶段模型中区分不同子模型的权重：

```python
def add_prefix_to_loaded_weights(weights: set[str], prefix: str) -> set[str]:
    """Add a prefix to the names of the loaded weights."""
    return {maybe_prefix(prefix, name) for name in weights}
```

例如将 `{"model.layers.0.weight"}` 加上前缀 `"thinker"` 后变为 `{"thinker.model.layers.0.weight"}`。

### split_list_into_ranges

将张量中的值按照固定间隔分到不同的桶中，这是一个优化版的分组工具：

```python
def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    if lst.numel() == 0:
        return []
    # Move to CPU and convert to list once (High Speedup)
    data_list = lst.detach().cpu().tolist()
    max_val = int(torch.max(lst).item())
    ranges: list[list[int]] = [[] for _ in range((max_val // interval) + 1)]
    for num in data_list:
        index = int(num // interval)
        ranges[index].append(num)
    return ranges
```

相比 `OmniMRotaryEmbedding._split_list_into_ranges`，该版本：
- 处理了空张量情况
- 先将张量转为 CPU 列表再遍历（注释中标注了 "High Speedup"），避免逐元素调用 `.item()` 的开销

### safe_tensor_reshape

安全的 reshape 操作，处理 `None` 输入：

```python
def safe_tensor_reshape(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Reshape a tensor safely."""
    if tensor is None:
        return None
    return tensor.reshape(shape)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `add_prefix_to_loaded_weights` | 函数 | 为权重名称集合添加前缀 |
| `split_list_into_ranges` | 函数 | 将值按间隔分桶（优化版） |
| `safe_tensor_reshape` | 函数 | None-safe 的 reshape |

## 与其他模块的关系

- **vllm.model_executor.models.utils.maybe_prefix**: 复用 vLLM 的前缀拼接工具
- **models/ 中的各模型**: `add_prefix_to_loaded_weights` 在多阶段模型加载权重时使用
- **layers/rotary_embedding/mrope.py**: `split_list_into_ranges` 与 MRoPE 中的同名静态方法功能相似，但此处是优化版

## 总结

`utils.py` 提供了三个简洁但实用的工具函数，主要服务于多阶段模型的权重管理和张量操作。`split_list_into_ranges` 的 CPU 预转换优化体现了对性能的关注。
