# layers/ -- 自定义神经网络层模块

## 文件概述

`layers/` 模块包含 vllm-omni 项目扩展的自定义神经网络层实现。当前主要包含对旋转位置编码（Rotary Embedding）的多模态扩展，以支持图像、视频、音频等多种模态的位置编码计算。

**目录路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/layers/`

## 模块结构

```
layers/
├── __init__.py                    # 空文件
└── rotary_embedding/
    ├── __init__.py                # 导出 OmniMRotaryEmbedding
    └── mrope.py                   # 多模态旋转位置编码实现
```

## 子模块导航

| 文件 | 说明 | 文档链接 |
|------|------|----------|
| `rotary_embedding/__init__.py` | 导出接口 | [rotary_embedding/init.md](rotary_embedding/init.md) |
| `rotary_embedding/mrope.py` | OmniMRotaryEmbedding 核心实现 | [rotary_embedding/mrope.md](rotary_embedding/mrope.md) |

## 与其他模块的关系

- **vllm 原生层**: `OmniMRotaryEmbedding` 继承自 vLLM 原生的 `MRotaryEmbedding`，在其基础上扩展多模态支持
- **models/**: 各个模型在计算注意力时使用这些自定义层来处理多模态输入的位置编码
- **stage_input_processors/**: 部分处理器在构建模型输入时会间接依赖位置编码的计算逻辑

## 总结

`layers/` 模块目前专注于多模态旋转位置编码的扩展实现，是 vllm-omni 支持多模态推理（图像、视频、音频混合输入）的关键底层组件。
