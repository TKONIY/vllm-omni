# model_executor/ -- 模型执行器模块总览

## 文件概述

`model_executor/` 是 vllm-omni 项目的核心模块之一，负责模型的加载、注册、执行以及多阶段流水线的配置与数据流转。该模块为各类多模态模型（语音、图像、视频等）提供统一的执行框架。

## 模块结构

```
model_executor/
├── __init__.py                  # 模块初始化（空文件）
├── custom_process_mixin.py      # 阶段前后处理 Mixin 基类
├── layers/                      # 自定义神经网络层
│   └── rotary_embedding/        # 多模态旋转位置编码扩展
│       ├── __init__.py
│       └── mrope.py             # OmniMRotaryEmbedding 实现
├── model_loader/                # 模型权重下载工具
│   ├── __init__.py
│   └── weight_utils.py          # HuggingFace 权重下载
├── models/                      # 模型定义与注册
│   ├── __init__.py              # 导出核心模型类
│   ├── output_templates.py      # 统一输出数据结构 OmniOutput
│   ├── registry.py              # 模型注册表 OmniModelRegistry
│   ├── utils.py                 # 模型工具函数
│   └── <各模型子目录>/           # 各模型具体实现
├── stage_configs/               # 多阶段流水线 YAML 配置
│   ├── __init__.py
│   └── *.yaml                   # 各模型的阶段配置文件
└── stage_input_processors/      # 阶段间数据转换处理器
    ├── __init__.py
    ├── bagel.py                 # Bagel CFG 扩展处理
    ├── chunk_size_utils.py      # 动态分块大小计算
    ├── cosyvoice3.py            # CosyVoice3 文本到语音
    ├── fish_speech.py           # Fish Speech 编解码
    ├── glm_image.py             # GLM-Image AR 到扩散
    ├── mammoth_moda2.py         # MammothModa2 AR 到 DiT
    ├── mimo_audio.py            # MiMo-Audio 编解码
    ├── qwen2_5_omni.py          # Qwen2.5-Omni 阶段转换
    ├── qwen3_omni.py            # Qwen3-Omni 阶段转换
    └── qwen3_tts.py             # Qwen3-TTS 阶段转换
```

## 核心设计理念

1. **多阶段流水线架构**: 通过 YAML 配置文件定义多阶段（Thinker/Talker/Code2Wav 等）推理流水线，每个阶段可以运行在不同 GPU 上
2. **统一模型注册**: `OmniModelRegistry` 合并 vLLM 原生模型与 Omni 扩展模型，提供统一的模型发现机制
3. **可插拔处理器**: `stage_input_processors` 提供各模型特化的阶段间数据转换逻辑，支持同步和异步流式处理
4. **Mixin 扩展模式**: `CustomProcessMixin` 允许在运行时动态注入前/后处理逻辑

## 子模块导航

| 子模块 | 说明 | 文档链接 |
|--------|------|----------|
| 核心文件 | 初始化与 Mixin 基类 | [custom_process_mixin.md](custom_process_mixin.md) |
| layers/ | 自定义网络层 | [layers/index.md](layers/index.md) |
| model_loader/ | 权重下载工具 | [model_loader/index.md](model_loader/index.md) |
| models/ | 模型注册与输出模板 | [models/index.md](models/index.md) |
| stage_configs/ | 流水线配置 | [stage_configs/index.md](stage_configs/index.md) |
| stage_input_processors/ | 阶段间处理器 | [stage_input_processors/index.md](stage_input_processors/index.md) |

## 与其他模块的关系

- **engine/**: 引擎模块调用 `stage_configs` 中的 YAML 配置来初始化多阶段流水线
- **worker/**: Worker 模块使用 `model_loader` 加载模型权重，使用 `models/registry.py` 查找模型类
- **core/sched/**: 调度器（如 `OmniARScheduler`）在 YAML 配置中被引用
- **inputs/**: `OmniTokensPrompt` 数据结构被 `stage_input_processors` 广泛使用

## 总结

`model_executor/` 是 vllm-omni 的模型执行核心，它通过 YAML 配置驱动的多阶段流水线架构，支持 10+ 种多模态模型的高效推理。模块采用注册表模式管理模型、Mixin 模式扩展处理逻辑、处理器模式转换阶段间数据，形成了灵活且可扩展的执行框架。
