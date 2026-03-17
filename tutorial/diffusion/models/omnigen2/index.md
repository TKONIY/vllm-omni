# OmniGen2 模型教程索引

## 模块概述

`omnigen2` 模块实现了 OmniGen2 图像生成模型，采用 Lumina2 风格的分阶段 Transformer 架构。支持文本到图像生成和图像编辑两种模式，使用 Qwen2.5-VL 作为视觉语言编码器。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 空的模块入口 |
| [`omnigen2_transformer.py`](omnigen2_transformer.py.md) | Transformer 模型（分阶段精化架构） |
| [`pipeline_omnigen2.py`](pipeline_omnigen2.py.md) | 生成/编辑管线 |

## 架构特点

- **分阶段精化**：文本 Refiner + 噪声 Refiner + 主 Transformer
- **视觉语言编码**：使用 Qwen2.5-VL 联合编码文本和图像
- **图像编辑**：通过 latent 拼接和条件掩码实现
- **自定义调度器**：轻量级 Flow Matching Euler 调度器
