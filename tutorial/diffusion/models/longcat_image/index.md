# LongCat Image 模型教程索引

## 模块概述

`longcat_image` 模块实现了 LongCat Image 的图像生成和图像编辑功能。模型基于 Flux 架构，采用双流/单流混合 Transformer 设计，集成了 Qwen2.5-VL 文本编码器，支持张量并行（TP）和序列并行（SP）的高效推理。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 模块入口，导出核心组件 |
| [`longcat_image_transformer.py`](longcat_image_transformer.py.md) | Transformer 模型实现，含注意力层、SP 支持 |
| [`pipeline_longcat_image.py`](pipeline_longcat_image.py.md) | 文本到图像生成管线 |
| [`pipeline_longcat_image_edit.py`](pipeline_longcat_image_edit.py.md) | 图像编辑管线 |

## 架构特点

- **双流/单流混合**：先通过双流 Transformer 块分别处理文本和图像，再通过单流块联合处理
- **序列并行**：通过 `_sp_plan` 声明式定义并行策略，图像序列分片、文本序列复制
- **张量并行**：使用 vLLM 的 QKVParallelLinear、RowParallelLinear 等层
- **提示词改写**：利用 Qwen2.5-VL 自动改写用户提示词
- **引号感知编码**：对引号内文本按字符编码，优化文字渲染效果
