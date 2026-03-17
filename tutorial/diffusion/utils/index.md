# utils/ — 工具函数子模块

## 模块概述

`utils/` 子模块提供了扩散模块使用的通用工具函数，包括 HuggingFace 模型检测、网络端口检测和 Transformer 配置处理。

## 文件索引

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口 |
| [`hf_utils.py`](hf_utils.md) | HuggingFace 扩散模型检测 |
| [`network_utils.py`](network_utils.md) | 网络端口可用性检测 |
| [`tf_utils.py`](tf_utils.md) | Transformer 配置参数提取 |

## 核心功能

| 工具 | 核心函数 | 用途 |
|------|----------|------|
| HF 检测 | `is_diffusion_model()` | 判断模型是否为扩散模型（三层回退策略） |
| 网络工具 | `is_port_available()` | 检测端口可用性（分布式初始化） |
| 配置工具 | `get_transformer_config_kwargs()` | 从 HF 配置提取模型构造参数（签名过滤） |
