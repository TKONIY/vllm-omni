# `__init__.py` — attention 模块初始化文件

## 文件概述

`__init__.py` 是 `vllm_omni/diffusion/attention/` 包的初始化文件。该文件内容极为简单，仅包含 Apache-2.0 许可证声明，不导出任何符号。

模块的实际功能通过其子模块提供：
- `layer.py`：核心 `Attention` 层
- `selector.py`：后端选择器
- `backends/`：各种注意力后端实现
- `parallel/`：并行注意力策略

## 关键代码解析

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

文件仅包含许可证头部，无任何导入或导出。使用者需要直接从子模块导入所需组件，例如：

```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.selector import get_attn_backend
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| （无） | — | 该文件不定义任何类或函数 |

## 与其他模块的关系

作为包的入口点，`__init__.py` 将 `attention/` 目录标记为一个 Python 包，使得其子模块（`layer`、`selector`、`backends`、`parallel`）可以被外部代码导入。

## 总结

这是一个纯粹的包标记文件，不包含业务逻辑。所有注意力相关的核心实现分布在该包的各个子模块中。
