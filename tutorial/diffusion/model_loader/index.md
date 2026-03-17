# 模型加载器子模块索引

## 概述

`model_loader/` 子模块负责扩散模型的权重加载，支持多种格式（safetensors、bin、pt、GGUF）、多种来源（本地、Hugging Face Hub）、多种部署模式（单 GPU、HSDP 多 GPU）以及与量化系统的集成。

## 架构设计

```
model_loader/
├── __init__.py            # 模块入口（当前为空）
├── diffusers_loader.py    # 核心加载器 DiffusersPipelineLoader
└── gguf_adapters/         # GGUF 格式权重名称映射适配器
    ├── __init__.py
    ├── base.py
    ├── flux2_klein.py
    └── z_image.py
```

## 文件列表

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 模块入口（空） | [__init__.md](./__init__.md) |
| `diffusers_loader.py` | 核心加载器：权重准备、迭代、GGUF/HSDP 加载、量化后处理 | [diffusers_loader.md](./diffusers_loader.md) |

## 加载流程

```
用户配置 (OmniDiffusionConfig)
    │
    ▼
DiffusersPipelineLoader.load_model()
    │
    ├─ initialize_model()        # 创建模型实例
    │
    ├─ 是否 GGUF？
    │   ├─ 是 → _load_weights_with_gguf()
    │   │       ├─ get_gguf_adapter() → 选择适配器
    │   │       └─ 回退到 HF 补充缺失权重
    │   └─ 否 → load_weights()
    │           └─ _get_weights_iterator() → safetensors/多线程
    │
    ├─ _process_weights_after_loading()  # FP8 在线量化等
    │
    └─ model.eval()
```

## 子目录

- [gguf_adapters/](./gguf_adapters/index.md) — GGUF 格式权重映射适配器
