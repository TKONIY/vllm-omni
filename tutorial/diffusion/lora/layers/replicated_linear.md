# `replicated_linear.py` — 复制式线性层 LoRA 封装

## 文件概述

`replicated_linear.py` 定义了 `DiffusionReplicatedLinearWithLoRA`，用于为扩散模型中的复制式线性层（ReplicatedLinear）添加 LoRA 支持。复制式线性层在每个 GPU 上持有完整的权重副本，不进行张量并行切分。

## 关键代码解析

```python
class DiffusionReplicatedLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    ReplicatedLinearWithLoRA,
):
    """Diffusion ReplicatedLinear with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA"""
    pass
```

与列并行和行并行变体一样，通过多重继承将 `DiffusionBaseLinearLayerWithLoRA` 的 `apply()` 方法优先于 vLLM 的 `ReplicatedLinearWithLoRA` 实现。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionReplicatedLinearWithLoRA` | 类 | 复制式线性层的 LoRA 封装，适用于不进行 TP 切分的层 |

## 与其他模块的关系

- **`base_linear.py`**：继承 `DiffusionBaseLinearLayerWithLoRA`，获得 torch matmul 的 `apply()` 实现。
- **vLLM `ReplicatedLinearWithLoRA`**：继承其层匹配和权重管理逻辑。
- **`../utils.py`**：在 `from_layer_diffusion` 的匹配链中优先级最低（最后尝试）。

## 总结

`DiffusionReplicatedLinearWithLoRA` 是 LoRA 层类型体系中处理非并行线性层的最后兜底选项。当线性层不属于列并行或行并行类型时，会使用此类进行封装。
