# `column_parallel_linear.py` — 列并行 LoRA 层实现

## 文件概述

`column_parallel_linear.py` 定义了四种列并行线性层的扩散模型 LoRA 封装类。这些类通过 Python 的多重继承（MRO）机制，将 `DiffusionBaseLinearLayerWithLoRA` 的 `apply()` 方法优先于 vLLM 原有实现，从而在列并行场景下使用 torch matmul 替代 punica_wrapper。

## 关键代码解析

### 1. 基础列并行 LoRA 层

```python
class DiffusionColumnParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
):
    """Diffusion ColumnParallelLinear with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA"""
    pass
```

通过将 `DiffusionBaseLinearLayerWithLoRA` 放在 MRO 前面，确保 `apply()` 使用扩散模型的 torch matmul 实现而非 vLLM 的 punica 实现。类本身无需额外代码。

### 2. 融合列并行 LoRA 层

```python
class DiffusionMergedColumnParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    MergedColumnParallelLinearWithLoRA,
):
    """Diffusion MergedColumnParallelLinear (gate_up_proj) with LoRA."""
    pass
```

用于处理融合的列并行投影（如 MLP 中的 `gate_up_proj`），支持多 slice LoRA。

### 3. QKV 并行 LoRA 层

```python
class DiffusionQKVParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    QKVParallelLinearWithLoRA,
):
    """Diffusion QKVParallelLinear with single LoRA."""
    pass
```

用于 QKV 投影使用单个 LoRA 权重的场景。

### 4. 融合 QKV 并行 LoRA 层

```python
class DiffusionMergedQKVParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    MergedQKVParallelLinearWithLoRA,
):
    """Diffusion MergedQKVParallelLinear (to_qkv) with 3 LoRAs."""
    pass
```

用于 QKV 投影使用三个独立 LoRA 权重的场景（Q、K、V 各一个）。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionColumnParallelLinearWithLoRA` | 类 | 标准列并行线性层的 LoRA 封装 |
| `DiffusionMergedColumnParallelLinearWithLoRA` | 类 | 融合列并行（如 gate_up_proj）的 LoRA 封装 |
| `DiffusionQKVParallelLinearWithLoRA` | 类 | QKV 并行（单 LoRA）的封装 |
| `DiffusionMergedQKVParallelLinearWithLoRA` | 类 | 融合 QKV（3 个 LoRA）的封装 |

## 与其他模块的关系

- **`base_linear.py`**：所有类继承 `DiffusionBaseLinearLayerWithLoRA`，获得 torch matmul 的 `apply()` 实现。
- **vLLM 层类**：每个类同时继承对应的 vLLM 列并行 LoRA 类，复用其 `can_replace_layer()`、权重管理和 TP 切分逻辑。
- **`../utils.py`**：`from_layer_diffusion` 函数按优先级尝试这些类进行层替换。

## 总结

本文件通过多重继承的 mixin 模式，以最小的代码量实现了四种列并行 LoRA 层。设计的关键在于 MRO 顺序：`DiffusionBaseLinearLayerWithLoRA` 排在前面，确保其 `apply()` 方法优先被调用，而 vLLM 原有类的 `can_replace_layer()` 等其他方法仍然正常工作。
