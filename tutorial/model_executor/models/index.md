# models/ -- 模型定义与注册模块

## 文件概述

`models/` 模块是 vllm-omni 的模型管理中心，包含模型注册表、统一输出数据结构、工具函数以及各模型的具体实现。本文档覆盖模块级的公共文件（`__init__.py`、`output_templates.py`、`registry.py`、`utils.py`），各模型子目录的详细实现不在本教程范围内。

**目录路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/models/`

## 模块结构

```
models/
├── __init__.py              # 导出核心模型类
├── output_templates.py      # OmniOutput 统一输出结构
├── registry.py              # OmniModelRegistry 模型注册表
├── utils.py                 # 通用工具函数
├── bagel/                   # Bagel 图像生成模型
├── cosyvoice3/              # CosyVoice3 语音合成模型
├── fish_speech/             # Fish Speech 语音合成模型
├── glm_image/               # GLM-Image 图像生成模型
├── hunyuan_image3/          # Hunyuan-Image3 图像生成模型
├── mammoth_moda2/           # MammothModa2 多模态生成模型
├── mimo_audio/              # MiMo-Audio 语音对话模型
├── qwen2_5_omni/            # Qwen2.5-Omni 全模态模型
├── qwen3_omni/              # Qwen3-Omni MoE 全模态模型
└── qwen3_tts/               # Qwen3-TTS 语音合成模型
```

## 子文件导航

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `__init__.py` | 模型导出 | [init.md](init.md) |
| `output_templates.py` | 统一输出结构 | [output_templates.md](output_templates.md) |
| `registry.py` | 模型注册表 | [registry.md](registry.md) |
| `utils.py` | 工具函数 | [utils.md](utils.md) |

## 与其他模块的关系

- **model_loader/**: 加载权重到此处定义的模型实例中
- **stage_configs/**: YAML 配置通过 `model_arch` 字段引用此处注册的模型架构名
- **engine/**: 引擎通过 `OmniModelRegistry` 查找并实例化模型
- **vllm.model_executor.models.registry**: `OmniModelRegistry` 合并了 vLLM 原生模型注册表

## 总结

`models/` 模块通过 `OmniModelRegistry` 统一管理 10+ 种多模态模型的注册与发现，通过 `OmniOutput` 提供标准化的输出接口，是 vllm-omni 模型管理的核心枢纽。
