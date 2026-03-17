# model_loader/ -- 模型权重加载模块

## 文件概述

`model_loader/` 模块提供模型权重的下载和加载工具函数，主要封装了从 HuggingFace Hub（或 ModelScope）下载指定模式权重文件的功能。

**目录路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/model_loader/`

## 模块结构

```
model_loader/
├── __init__.py       # 空文件
└── weight_utils.py   # 权重下载工具函数
```

## 子模块导航

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `weight_utils.py` | HuggingFace 权重下载工具 | [weight_utils.md](weight_utils.md) |

## 与其他模块的关系

- **vllm.model_executor.model_loader.weight_utils**: 复用 vLLM 原生的 `DisabledTqdm` 和 `get_lock`
- **worker/**: Worker 在初始化模型时调用此模块下载权重
- **models/**: 各模型可能通过此模块下载额外的权重文件（如子模型权重）

## 总结

`model_loader/` 是一个轻量的工具模块，提供了支持多种匹配模式的 HuggingFace 权重下载功能，是模型初始化流程中的重要环节。
