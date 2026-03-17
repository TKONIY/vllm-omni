# `autoencoders/__init__.py` -- 分布式自编码器子模块入口

## 文件概述

`autoencoders/__init__.py` 是分布式 VAE 自编码器子模块的入口文件。该文件为空（仅包含空行），表明该目录作为 Python 包存在，各分布式 VAE 实现通过直接导入具体文件使用。

## 与其他模块的关系

- **autoencoder_kl.py**: 标准 AutoencoderKL 的分布式版本
- **autoencoder_kl_qwenimage.py**: Qwen 图像模型 VAE 的分布式版本
- **autoencoder_kl_wan.py**: Wan 视频模型 VAE 的分布式版本
- **distributed_vae_executor.py**: 核心分布式执行框架

## 总结

该入口文件仅标识目录为 Python 包，所有分布式 VAE 功能在各自的文件中实现。
