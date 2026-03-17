# LoRA 层类型索引

## 概述

`lora/layers/` 子目录实现了扩散模型专用的 LoRA 层类型。所有类型继承自 `DiffusionBaseLinearLayerWithLoRA`，通过多重继承的 mixin 模式将扩散模型的 torch matmul `apply()` 方法与 vLLM 原有各类线性层的权重管理和 TP 切分逻辑组合在一起。

## 架构设计

```
layers/
├── __init__.py                    # 统一导出所有层类型
├── base_linear.py                 # 基类：torch matmul apply() + 属性转发
├── column_parallel_linear.py      # 列并行：Column、MergedColumn、QKV、MergedQKV
├── replicated_linear.py           # 复制式线性层
└── row_parallel_linear.py         # 行并行线性层
```

## 文件列表

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 统一导出 7 种 LoRA 层类型 | [__init__.md](./__init__.md) |
| `base_linear.py` | 基类：torch matmul 替代 punica，slice 跟踪，属性转发 | [base_linear.md](./base_linear.md) |
| `column_parallel_linear.py` | 4 种列并行 LoRA 层：Column、MergedColumn、QKV、MergedQKV | [column_parallel_linear.md](./column_parallel_linear.md) |
| `replicated_linear.py` | 复制式线性层 LoRA 封装 | [replicated_linear.md](./replicated_linear.md) |
| `row_parallel_linear.py` | 行并行线性层 LoRA 封装 | [row_parallel_linear.md](./row_parallel_linear.md) |

## 层类型对照

| Diffusion LoRA 类 | vLLM 原始类 | 典型用途 |
|-------------------|-------------|----------|
| `DiffusionColumnParallelLinearWithLoRA` | `ColumnParallelLinearWithLoRA` | 标准列并行投影 |
| `DiffusionMergedColumnParallelLinearWithLoRA` | `MergedColumnParallelLinearWithLoRA` | gate_up_proj 融合投影 |
| `DiffusionQKVParallelLinearWithLoRA` | `QKVParallelLinearWithLoRA` | QKV 并行（单 LoRA） |
| `DiffusionMergedQKVParallelLinearWithLoRA` | `MergedQKVParallelLinearWithLoRA` | to_qkv 融合（3 LoRA） |
| `DiffusionRowParallelLinearWithLoRA` | `RowParallelLinearWithLoRA` | 输出/下投影 |
| `DiffusionReplicatedLinearWithLoRA` | `ReplicatedLinearWithLoRA` | 非并行线性层 |

## 设计要点

- **MRO 优先级**：`DiffusionBaseLinearLayerWithLoRA` 在 MRO 中排在 vLLM 类之前，确保 `apply()` 使用 torch matmul。
- **最小代码原则**：子类均为空类（`pass`），所有逻辑由基类和 vLLM 父类提供。
- **匹配优先级**：`from_layer_diffusion` 中按从具体到通用的顺序尝试匹配。
