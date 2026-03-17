# `backends/__init__.py` — 注意力后端包初始化

## 文件概述

`backends/__init__.py` 是 `attention/backends/` 包的初始化文件，仅包含 Apache-2.0 许可证声明，不导出任何符号。

## 关键代码解析

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

该文件将 `backends/` 目录标记为 Python 包。各后端实现通过各自的模块文件提供：

- `abstract.py`：抽象基类定义
- `flash_attn.py`：Flash Attention 后端
- `sdpa.py`：PyTorch SDPA 后端
- `sage_attn.py`：Sage Attention 后端
- `registry.py`：后端注册表
- `ring_flash_attn.py`：Ring Flash Attention 实现
- `ring_pytorch_attn.py`：Ring PyTorch Attention 实现

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| （无） | — | 该文件不定义任何类或函数 |

## 与其他模块的关系

作为包入口点，允许外部代码通过 `vllm_omni.diffusion.attention.backends.xxx` 路径导入各后端实现。

## 总结

纯粹的包标记文件，后端的实际实现分布在各子模块中。
