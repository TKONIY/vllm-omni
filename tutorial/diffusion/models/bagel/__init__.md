# `bagel/__init__.py` -- Bagel 模型包初始化

## 文件概述

Bagel 模型子包的入口文件。该文件为空（仅包含许可证声明），说明 Bagel 相关类的导入由使用方直接从子模块导入。

**文件路径**: `vllm_omni/diffusion/models/bagel/__init__.py`

## 与其他模块的关系

Bagel 子包包含三个核心模块：
- `autoencoder.py`: 自编码器（VAE），用于图像编解码
- `bagel_transformer.py`: Bagel Mixture-of-Tokens Transformer 核心模型
- `pipeline_bagel.py`: 完整的 Bagel 生成管线

## 总结

Bagel 是一个基于 Mixture-of-Tokens (MoT) 架构的多模态生成模型，支持文本到图像和图像编辑任务。
