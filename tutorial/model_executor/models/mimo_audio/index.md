# MiMo-Audio 模型模块架构概览

## 模块简介

MiMo-Audio 是小米的多模态音频理解与生成模型，基于 Qwen2 LLM 扩展了多通道音频编解码能力。采用融合 Thinker-Talker 架构，在单一 LLM 中同时完成文本理解和语音 token 生成。使用自研的 MiMoAudioTokenizer 进行音频编解码。

## 架构图

```
文本 + 音频输入
       │
       ▼
┌─────────────────────────────────────┐
│  MiMoAudioLLMMultiModalProcessor   │  ← 多模态处理
│  MiMoAudioDataParser               │  ← 音频→codec codes
│  (MiMoAudioTokenizerWorker)         │  ← ONNX编码
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  MiMoAudioForConditionalGeneration  │  ← 顶层路由
│  ├── fused_thinker_talker 阶段      │
│  │   └── MiMoAudioLLMModel          │
│  │       ├── Qwen2Model (backbone)  │  ← vLLM PagedAttention
│  │       ├── speech_embeddings       │  ← 多通道语音嵌入
│  │       ├── input_local_transformer │  ← 音频输入局部处理
│  │       ├── local_transformer       │  ← 多通道局部预测
│  │       └── speech_group_downcast   │  ← 维度投影
│  └── code2wav 阶段                   │
│      └── MiMoAudioToken2Wav          │
│          └── MiMoAudioTokenizerWorker│  ← VQ解码→波形
│              └── MiMoAudioTokenizer  │
│                  ├── AudioEncoder    │  ← RVQ编码器
│                  └── AudioDecoder    │  ← Vocos解码器
│                      └── TransformerVocos │ ← ISTFT合成
└─────────────────────────────────────┘
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 空模块入口 |
| `config_mimo_audio.py` | MiMo-Audio 配置类（LLM + Tokenizer） |
| `mimo_audio.py` | 顶层模型：多模态处理器 + 阶段路由 |
| `mimo_audio_code2wav.py` | Code2Wav：codec 解码 + 流式音频生成 |
| `mimo_audio_llm.py` | LLM 模型：Qwen2 + 多通道音频 MTP |
| `modeling_audio_tokenizer.py` | 音频 tokenizer：编码器 + 解码器 + Vocos |
| `modeling_rope_utils.py` | RoPE 工具：多种缩放策略实现 |
| `quantization.py` | 残差向量量化（RVQ）实现 |

## 核心设计思想

1. **融合 Thinker-Talker**：文本理解和语音 token 生成在同一 LLM 中完成，通过交错（interleave）机制在文本 token 间插入音频 codec token。

2. **多通道音频编码**：使用 8 通道 codec 编码，每 4 帧（group_size=4）组合为一个 token group，实现高压缩比音频表示。

3. **Delay Pattern**：8 个 codec 通道使用 `0-1-2-3-4-5-6-7` 的延迟模式，确保自回归生成时的因果性。

4. **流式解码**：Code2Wav 阶段支持分块解码（chunk + left context），通过裁剪左上下文实现无缝拼接。

5. **自研 Tokenizer**：MiMoAudioTokenizer 使用 Transformer 编码器 + RVQ 量化 + Vocos 解码器的架构，支持 24kHz 音频的高质量编解码。
