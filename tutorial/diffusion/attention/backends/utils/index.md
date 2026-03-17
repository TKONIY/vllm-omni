# utils/ — 注意力后端工具函数

## 模块概述

`utils/` 包含注意力后端共用的工具函数，主要围绕 Flash Attention 的导入管理和变长序列（varlen）的 unpad/pad 操作。

## 模块结构

```
utils/
├── __init__.py   # 导出 _pad_input, _unpad_input, _upad_input
└── fa.py         # Flash Attention 导入管理 & unpad/pad 工具
```

## 核心功能

### 1. Flash Attention 导入管理

`fa.py` 通过回退链自动检测和导入可用的 Flash Attention 版本：

| 平台 | 导入优先级 |
|------|-----------|
| CUDA | FA3 (fa3_fwd_interface) → FA3 (flash_attn_interface) → FA2 (flash_attn) → FA2 (flash_attn.flash_attn_interface) |
| ROCm | Aiter (aiter) |
| XPU | vllm 内置 fa_utils |

### 2. Unpad/Pad 工具

用于 Flash Attention 的变长序列处理：

| 函数 | 说明 |
|------|------|
| `_unpad_input` | 根据掩码移除 padding → 紧凑格式 `(total_nnz, ...)` |
| `_pad_input` | 紧凑格式 → 恢复 padding `(batch, seqlen, ...)` |
| `_upad_input` | 统一 Q/K/V 的 unpad，避免重复计算 |

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](./__init__.md) | 导出核心工具函数 |
| [`fa.py`](./fa.md) | Flash Attention 导入回退链 & unpad/pad 工具 |
