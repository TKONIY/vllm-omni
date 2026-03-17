# backends/ — 注意力计算后端

## 模块概述

`backends/` 包含所有注意力计算后端的实现。每个后端由一个工厂类（`AttentionBackend`）和一个实现类（`AttentionImpl`）组成，分别负责能力声明和具体计算。后端通过注册表系统管理，支持运行时覆盖。

## 架构层次

```
backends/
├── abstract.py          # 抽象基类：AttentionBackend / AttentionImpl / AttentionMetadata
├── registry.py          # 后端注册表枚举（支持运行时覆盖）
├── flash_attn.py        # Flash Attention 后端（CUDA / XPU / NPU）
├── sdpa.py              # PyTorch SDPA 后端（全平台通用）
├── sage_attn.py         # Sage Attention 后端（仅 CUDA）
├── ring_flash_attn.py   # Ring Attention + Flash Attn 内核
├── ring_pytorch_attn.py # Ring Attention + PyTorch SDPA 内核
├── ring/                # Ring Attention 底层组件
└── utils/               # 工具函数（FA 导入管理、unpad/pad）
```

## 后端对比

| 后端 | 支持平台 | 支持掩码 | 支持 float32 | Head Sizes |
|------|----------|----------|-------------|------------|
| Flash Attention | CUDA, XPU, NPU | 是 | 否 | 64, 96, 128, 192, 256 |
| SDPA | CUDA, XPU, HIP, NPU | 是 | 是 | 1-1023 |
| Sage Attention | CUDA | 否 | 否 | 32, 64, 96, 128, 160, 192, 224, 256 |

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](./__init__.md) | 包初始化 |
| [`abstract.py`](./abstract.md) | 抽象基类与元数据定义 |
| [`registry.py`](./registry.md) | 后端注册表（枚举 + 覆盖机制） |
| [`flash_attn.py`](./flash_attn.md) | Flash Attention 后端 |
| [`sdpa.py`](./sdpa.md) | PyTorch SDPA 后端 |
| [`sage_attn.py`](./sage_attn.md) | Sage Attention 后端 |
| [`ring_flash_attn.py`](./ring_flash_attn.md) | Ring Flash Attention 实现 |
| [`ring_pytorch_attn.py`](./ring_pytorch_attn.md) | Ring PyTorch Attention 实现 |

### 子目录

| 目录 | 说明 |
|------|------|
| [`ring/`](./ring/index.md) | Ring Attention 底层组件（内核、选择器、工具） |
| [`utils/`](./utils/index.md) | Flash Attention 工具函数 |

## 扩展指南

添加新后端需要：
1. 创建新文件，实现 `AttentionBackend` 和 `AttentionImpl` 子类
2. 在 `registry.py` 的 `DiffusionAttentionBackendEnum` 中添加枚举成员
3. 在平台层配置新后端的选择逻辑
