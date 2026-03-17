# `utils/__init__.py` — 后端工具包初始化

## 文件概述

`utils/__init__.py` 是 `backends/utils/` 包的初始化文件，导出了 Flash Attention 相关的 unpad/pad 工具函数。

## 关键代码解析

```python
from vllm_omni.diffusion.attention.backends.utils.fa import _pad_input, _unpad_input, _upad_input

__all__ = [
    "_pad_input",
    "_unpad_input",
    "_upad_input",
]
```

导出三个核心工具函数：
- `_pad_input`：将 unpadded 的紧凑张量还原为带 padding 的标准形状
- `_unpad_input`：根据注意力掩码移除 padding，生成紧凑张量
- `_upad_input`：统一的 Q/K/V unpad 函数，避免重复计算中间结果

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_pad_input` | 函数 | 紧凑张量 → 带 padding 张量 |
| `_unpad_input` | 函数 | 带 padding 张量 → 紧凑张量 |
| `_upad_input` | 函数 | Q/K/V 统一 unpad 函数 |

## 与其他模块的关系

- **`fa.py`**：所有函数均来自此模块
- **`flash_attn.py`**：`FlashAttentionImpl._forward_varlen_masked` 使用这些工具

## 总结

该文件是 `fa.py` 中关键 unpad/pad 函数的统一导出入口，方便外部模块引用。
