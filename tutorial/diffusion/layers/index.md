# layers/ — 多平台自定义算子层

## 模块概述

`layers/` 子模块提供了扩散 Transformer 模型使用的基础算子层，包括多平台调度基类、自适应层归一化和旋转位置编码。所有层都继承 `CustomOp` 基类，自动适配 CUDA、ROCm、NPU 和 XPU 等硬件平台。

## 架构设计

```
CustomOp (平台调度基类)
  ├── AdaLayerNorm (自适应层归一化)
  │     └── 支持 CFG 条件分支、MindIE 融合算子
  └── RotaryEmbedding (旋转位置编码)
        └── 支持 GPT-NeoX/GPT-J 风格、多种 kernel
```

## 文件索引

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口 |
| [`custom_op.py`](custom_op.md) | 多平台算子调度基类 |
| [`adalayernorm.py`](adalayernorm.md) | 自适应层归一化（AdaLayerNorm） |
| [`rope.py`](rope.md) | 旋转位置编码（RoPE） |

## 核心设计

- **零开销调度**：`CustomOp` 在 `__init__` 时一次性选择平台实现，运行时直接调用缓存的方法
- **HIP 回退**：ROCm 的 `forward_hip` 默认回退到 `forward_cuda`，因为 HIP 通常兼容 CUDA
- **NPU 优化**：NPU 平台优先使用 MindIE 融合算子（`mindiesd`），回退到 torch_npu 或原生实现
- **多实现策略**：RoPE 在不同平台使用不同 kernel（vllm_flash_attn、flash_attn triton、mindiesd）
