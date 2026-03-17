# `__init__.py` -- 扩散模型包初始化

## 文件概述

`__init__.py` 是 `vllm_omni/diffusion/models/` 包的入口文件。它作为模块标识文件，仅包含一行文档字符串，声明本包包含扩散模型的各种实现。

**文件路径**: `vllm_omni/diffusion/models/__init__.py`

## 关键代码解析

```python
"""Diffusion model implementations."""
```

该文件非常简洁，仅通过文档字符串说明包的用途。实际的模型注册和导入通过各个子模块的 `__init__.py` 完成（如 `flux/__init__.py`、`sd3/__init__.py` 等）。

## 核心类/函数

本文件无导出类或函数。

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 子包 | `flux/`, `flux2/`, `sd3/`, `bagel/` 等 | 各个具体的扩散模型实现 |
| 工具文件 | `interface.py`, `progress_bar.py` | 模型接口协议和进度条混入类 |
| 子包 | `schedulers/` | 噪声调度器实现 |

## 总结

此文件是 Python 包的标准入口，声明 `diffusion/models/` 为一个 Python 包。所有具体的模型实现分布在各子目录中，各自通过独立的 `__init__.py` 进行模块导出。
