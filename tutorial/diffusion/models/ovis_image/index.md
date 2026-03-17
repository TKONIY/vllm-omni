# Ovis Image 模型教程索引

## 模块概述

`ovis_image` 模块实现了阿里巴巴 AIDC 的 Ovis Image 7B 图像生成模型。模型基于 Flux 架构的双流/单流混合 Transformer，使用 Qwen3 基座模型作为文本编码器。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块入口，导出核心组件 |
| [`ovis_image_transformer.py`](ovis_image_transformer.py.md) | Transformer 模型（SwiGLU FFN + 复数 RoPE） |
| [`pipeline_ovis_image.py`](pipeline_ovis_image.py.md) | 文本到图像生成管线 |

## 架构特点

- **Qwen3 文本编码**：使用 Qwen3 基座模型（非 VL），通过系统提示引导
- **复数 RoPE**：位置编码使用复数形式而非实数形式
- **SwiGLU FFN**：双流块使用 SwiGLU 激活函数
- **SiLU 门控 MLP**：单流块使用 SiLU 门控
- **文本 RMSNorm**：文本嵌入额外经过 RMSNorm 归一化
- **CFG 并行**：通过 CFGParallelMixin 支持分布式 CFG
