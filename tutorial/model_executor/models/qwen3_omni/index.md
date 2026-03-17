# Qwen3-Omni MoE 模型架构概述

## 目录结构

```
qwen3_omni/
├── __init__.py                              # 模块初始化，导出统一入口类
├── qwen3_moe.py                             # MoE 稀疏专家模型（Talker 使用）
├── qwen3_omni.py                            # 统一入口模型（协调 thinker/talker/code2wav）
├── qwen3_omni_code2wav.py                   # Code2Wav 模型（codec → 音频波形）
├── qwen3_omni_moe_code_predictor_mtp.py     # Code Predictor（预测残差 RVQ 层）
├── qwen3_omni_moe_talker.py                 # Talker 模型（文本 → codec codes）
└── qwen3_omni_moe_thinker.py                # Thinker 模型（多模态理解 + 文本生成）
```

## 整体架构

Qwen3-Omni MoE 是 Qwen2.5-Omni 的升级版，采用 **MoE（混合专家）架构** 和 **多层 RVQ** 编码：

```
输入(文本/图像/视频/音频)
        │
        ▼
  ┌──────────────────┐
  │     Thinker      │  阶段1: MoE Transformer + 多模态理解
  │   (MoE-30B/3B)   │  支持 MRoPE + DeepStack 视觉特征
  └──────┬───────────┘
         │ text hidden states + 中间层隐藏状态
         ▼
  ┌──────────────────┐
  │     Talker       │  阶段2: MoE Transformer → Layer-0 codec
  │  + Code Predictor│  Code Predictor: 预测 Layer 1~15 (MTP)
  └──────┬───────────┘
         │ 16层 RVQ codes
         ▼
  ┌──────────────────┐
  │    Code2Wav      │  阶段3: RVQ codes → 音频波形
  │  (Conv + Attn)   │  ~1280x 上采样
  └──────────────────┘
```

## 与 Qwen2.5-Omni 的主要区别

| 特性 | Qwen2.5-Omni | Qwen3-Omni MoE |
|------|-------------|----------------|
| 架构 | Dense Transformer | MoE (Sparse) |
| RVQ 层数 | 1 层 codec | 16 层 RVQ |
| Code Predictor | 无 | 自回归预测残差层 |
| 语音合成 | DiT + BigVGAN | Conv + Transformer + 上采样 |
| 视觉特征 | 单尺度 | DeepStack 多尺度 |
| Token 抑制 | bad_word_processor | GPU 端布尔掩码 |
| 流式支持 | 分块 ODE | 异步分块解码 |

## 核心设计特点

1. **MoE 稀疏激活**: Thinker 和 Talker 均使用 MoE 层，30B 参数中仅激活 3B
2. **多层 RVQ**: Talker 生成第 0 层 codec，Code Predictor 通过 MTP（多 token 预测）自回归生成剩余 15 层
3. **DeepStack**: 视觉编码器提取多尺度特征，通过额外的 merger 注入到不同的 Transformer 层
4. **异步流式**: 支持异步分块音频生成，减少首包延迟
5. **GPU 端 Token 抑制**: 使用预计算的布尔掩码在 GPU 上高效抑制非法 token

## 数据流说明

| 阶段 | 输入 | 输出 | 关键类 |
|------|------|------|--------|
| Thinker | input_ids + 多模态特征 | hidden states + 中间层状态 | `Qwen3OmniMoeThinkerForConditionalGeneration` |
| Talker | thinker embeddings | Layer-0 codec + talker hidden | `Qwen3OmniMoeTalkerForConditionalGeneration` |
| Code Predictor | Layer-0 codec + talker hidden | 16层 RVQ codes | `Qwen3OmniMoeTalkerCodePredictor` |
| Code2Wav | 16层 RVQ codes | 音频波形 | `Qwen3OmniMoeCode2Wav` |
| 统一入口 | 所有输入 | `OmniOutput` | `Qwen3OmniMoeForConditionalGeneration` |
