# CosyVoice3 模型模块架构概览

## 模块简介

CosyVoice3 是阿里巴巴 FunAudioLLM 团队的第三代语音合成模型，实现文本到语音（TTS）的高质量生成。该模型采用两阶段架构：

1. **Talker 阶段**：基于 Qwen2 LLM 的自回归语音 token 生成
2. **Code2Wav 阶段**：基于条件流匹配（CFM）+ HiFiGAN 的 token 到波形转换

## 架构图

```
文本 + 参考音频
       │
       ▼
┌─────────────────────────────┐
│  CosyVoice3MultiModalProcessor │  ← 音频特征提取、ONNX tokenizer
│  (speech_tokenizer, campplus)   │
└──────────┬──────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  CosyVoice3Model (talker)    │
│  ├── CosyVoice3LM            │  ← 语音 token 语言模型
│  │   ├── speech_embedding     │  ← 语音 token 嵌入
│  │   ├── VLLMQwen2Encoder     │  ← Qwen2 + PagedAttention
│  │   └── llm_decoder          │  ← 输出投影头
│  └── embed_input_ids()        │  ← 拼接 SOS + 文本 + TaskID + 音频
└──────────┬──────────────────┘
           │ 语音 token 序列
           ▼
┌──────────────────────────────┐
│  CosyVoice3Model (code2wav)  │
│  ├── CosyVoice3Code2Wav      │
│  │   ├── CausalMaskedDiffWithDiT │ ← 流匹配解码器
│  │   │   ├── input_embedding  │  ← token → 特征嵌入
│  │   │   ├── PreLookaheadLayer│  ← 前瞻卷积层
│  │   │   └── CausalConditionalCFM │ ← 条件流匹配
│  │   │       └── DiT estimator│  ← 扩散 Transformer
│  │   └── CausalHiFTGenerator  │  ← HiFiGAN 声码器
│  │       ├── F0 预测器         │
│  │       ├── NSF 正弦源模块    │
│  │       └── ISTFT 合成       │
│  └── hift                     │
└──────────────────────────────┘
           │
           ▼
      音频波形输出
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 空模块入口 |
| `config.py` | CosyVoice3 配置类，包含 LLM、Flow、HiFT 全部参数 |
| `cosyvoice3.py` | 顶层模型类，路由 talker/code2wav 阶段 |
| `cosyvoice3_code2wav.py` | Code2Wav 阶段：流匹配 + HiFiGAN 声码器 |
| `cosyvoice3_talker.py` | Talker 阶段：Qwen2 LM + 语音 embedding |
| `tokenizer.py` | CosyVoice2/3 专用 tokenizer（Qwen 基础 + 特殊 token）|
| `utils.py` | 工具函数：mel 频谱、音频加载、token 提取 |
| `code2wav_core/__init__.py` | Code2Wav 核心子模块入口 |
| `code2wav_core/cfm.py` | 条件流匹配（CFM）实现 |
| `code2wav_core/hifigan.py` | HiFiGAN 声码器及因果变体 |
| `code2wav_core/layers.py` | PreLookahead 卷积层 |

## 核心设计思想

1. **两阶段解耦**：Talker 负责文本到语音 token 的映射，Code2Wav 负责 token 到波形的合成，两者通过 `model_stage` 参数在同一 `CosyVoice3Model` 中切换。

2. **vLLM 集成**：Talker 阶段使用 vLLM 的 `Qwen2Model` + PagedAttention 管理 KV cache，实现高吞吐推理。

3. **因果流匹配**：Code2Wav 使用 `CausalConditionalCFM` 实现因果推理，支持流式音频生成。

4. **说话人克隆**：通过 CamPPlus 提取说话人嵌入 + ONNX 语音 tokenizer 提取参考音频 token，实现零样本语音克隆。
