# `layers/__init__.py` — LoRA 层类型导出

## 文件概述

`layers/__init__.py` 是 LoRA 层子模块的入口文件，负责从各个具体实现文件中导出所有扩散模型专用的 LoRA 层类型。

## 关键代码解析

```python
from .base_linear import DiffusionBaseLinearLayerWithLoRA
from .column_parallel_linear import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
)
from .replicated_linear import DiffusionReplicatedLinearWithLoRA
from .row_parallel_linear import DiffusionRowParallelLinearWithLoRA
```

导出了 7 个 LoRA 层类，覆盖扩散模型中所有可能遇到的线性层类型。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionBaseLinearLayerWithLoRA` | 类 | 基础 LoRA 层，使用 torch matmul 替代 punica_wrapper |
| `DiffusionColumnParallelLinearWithLoRA` | 类 | 列并行线性层的 LoRA 封装 |
| `DiffusionMergedColumnParallelLinearWithLoRA` | 类 | 融合列并行线性层（如 gate_up_proj）的 LoRA 封装 |
| `DiffusionQKVParallelLinearWithLoRA` | 类 | QKV 并行线性层的单 LoRA 封装 |
| `DiffusionMergedQKVParallelLinearWithLoRA` | 类 | 融合 QKV 并行线性层的多 LoRA 封装 |
| `DiffusionReplicatedLinearWithLoRA` | 类 | 复制式线性层的 LoRA 封装 |
| `DiffusionRowParallelLinearWithLoRA` | 类 | 行并行线性层的 LoRA 封装 |

## 与其他模块的关系

- **`base_linear.py`**：基类实现，所有层类型都继承自它。
- **`column_parallel_linear.py`**：列并行相关的 4 个层类型。
- **`replicated_linear.py`**：复制式线性层类型。
- **`row_parallel_linear.py`**：行并行线性层类型。
- **`../utils.py`**：`from_layer_diffusion` 函数使用这些导出类进行层替换。

## 总结

此文件作为统一入口，将分散在各文件中的 7 种 LoRA 层类型汇聚在一起，方便其他模块（如 `utils.py` 中的 `from_layer_diffusion`）导入使用。
