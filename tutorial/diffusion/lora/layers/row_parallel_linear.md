# `row_parallel_linear.py` — 行并行线性层 LoRA 封装

## 文件概述

`row_parallel_linear.py` 定义了 `DiffusionRowParallelLinearWithLoRA`，为扩散模型中的行并行线性层添加 LoRA 支持。行并行线性层将权重按行切分到不同 GPU 上，通常用于注意力机制的输出投影和 MLP 的下投影。

## 关键代码解析

```python
class DiffusionRowParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    RowParallelLinearWithLoRA,
):
    """Diffusion RowParallelLinear with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA"""
    pass
```

延续了统一的 mixin 模式：`DiffusionBaseLinearLayerWithLoRA` 在 MRO 中优先，确保使用 torch matmul 的 `apply()` 实现。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionRowParallelLinearWithLoRA` | 类 | 行并行线性层的 LoRA 封装 |

## 与其他模块的关系

- **`base_linear.py`**：继承 `DiffusionBaseLinearLayerWithLoRA`，获得 torch matmul 的 `apply()` 实现。
- **vLLM `RowParallelLinearWithLoRA`**：继承其层匹配和 TP 切分逻辑。
- **`../utils.py`**：在 `from_layer_diffusion` 中被尝试用于匹配行并行线性层。

## 总结

`DiffusionRowParallelLinearWithLoRA` 处理扩散模型中行并行切分的线性层。与其他 LoRA 层类型一样，它通过多重继承最小化了代码量，核心逻辑完全由基类 `DiffusionBaseLinearLayerWithLoRA` 提供。
