# Z-Image 模型教程索引

## 模块概述

`z_image` 模块实现了阿里巴巴的 Z-Image 图像生成模型，是本项目中并行化最为完整的模型之一。同时支持张量并行（TP）、序列并行（SP）和量化推理，使用分布式 VAE 进行高效解码。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 空的模块入口 |
| [`z_image_transformer.py`](z_image_transformer.py.md) | Transformer 模型（TP + SP + 量化） |
| [`pipeline_z_image.py`](pipeline_z_image.py.md) | 图像生成管线（分布式 VAE + CFG 截断） |

## 架构特点

- **完整并行化**：同时支持 TP、SP 和量化
- **TP 约束验证**：初始化时严格验证 TP 兼容性并提示支持的配置
- **分布式 VAE**：使用 `DistributedAutoencoderKL` 多 GPU 解码
- **序列并行**：通过 `UnifiedPrepare` 模块实现 SP 分片
- **多 patch size**：通过 ModuleDict 支持不同的 patch 配置
- **CFG 截断**：后期步骤可自动关闭 CFG
- **CFG 归一化**：防止 CFG 过度放大预测向量
- **list 接口**：Transformer 使用 list 输入/输出，支持变长序列
