# Qwen2.5-Omni 模型架构概述

## 目录结构

```
qwen2_5_omni/
├── __init__.py                    # 模块初始化（空文件）
├── audio_length.py                # 音频长度对齐工具函数
├── qwen2_5_omni.py                # 统一入口模型（协调 thinker/talker/token2wav）
├── qwen2_5_omni_talker.py         # Talker 模型（文本→语音编码）
├── qwen2_5_omni_thinker.py        # Thinker 模型（多模态理解→文本）
├── qwen2_5_omni_token2wav.py      # Token2Wav 模型（编码→梅尔频谱→波形）
└── qwen2_old.py                   # 旧版 Qwen2 因果语言模型（Talker 内部使用）
```

## 整体架构

Qwen2.5-Omni 采用 **三阶段流水线** 架构，将多模态理解与语音合成解耦：

```
输入(文本/图像/视频/音频)
        │
        ▼
  ┌──────────────┐
  │   Thinker    │  阶段1: 多模态理解 + 文本生成
  │  (Stage 0)   │  支持 MRoPE 位置编码
  └──────┬───────┘
         │ 文本 hidden states + embeddings
         ▼
  ┌──────────────┐
  │   Talker     │  阶段2: 文本嵌入 → RVQ codec codes
  │  (Stage 1)   │  使用旧版 Qwen2 解码器
  └──────┬───────┘
         │ codec tokens
         ▼
  ┌──────────────┐
  │  Token2Wav   │  阶段3: codec → 梅尔频谱 → 音频波形
  │  (Stage 2)   │  DiT + BigVGAN 声码器
  └──────────────┘
```

## 核心设计特点

1. **阶段分离**: 每个阶段作为独立的 vLLM 模型实例运行，可分配到不同设备
2. **MRoPE 位置编码**: Thinker 使用多维旋转位置编码，支持图像/视频/音频的空间和时间维度
3. **流式音频生成**: Token2Wav 支持分块 ODE 求解和流式声码器输出
4. **权重映射**: 使用 `WeightsMapper` 将 HuggingFace 权重名映射到 vLLM 内部结构

## 数据流说明

| 阶段 | 输入 | 输出 | 关键类 |
|------|------|------|--------|
| Thinker | input_ids + 多模态特征 | text hidden states | `Qwen2_5OmniThinkerForConditionalGeneration` |
| Talker | thinker embeddings + codec tokens | codec logits / hidden states | `Qwen2_5OmniTalkerForConditionalGeneration` |
| Token2Wav | codec codes + speaker embedding | 音频波形 | `Qwen2_5OmniToken2WavForConditionalGenerationVLLM` |
| 统一入口 | 所有输入 | `OmniOutput` | `Qwen2_5OmniForConditionalGeneration` |

## 模块依赖关系

- `qwen2_5_omni.py` 依赖所有其他模块，是统一入口
- `qwen2_5_omni_thinker.py` 继承 vLLM 上游的 Qwen2.5-Omni Thinker mixin
- `qwen2_5_omni_talker.py` 使用 `qwen2_old.py` 中的 Qwen2 因果语言模型
- `qwen2_5_omni_token2wav.py` 独立实现 DiT + BigVGAN 架构
- `audio_length.py` 为 token2wav 提供纯 Python 的长度对齐工具
