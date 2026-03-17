# Qwen3-TTS 模型架构概述

## 目录结构

```
qwen3_tts/
├── __init__.py                           # 模块初始化（空文件）
├── configuration_qwen3_tts.py            # 模型配置类定义
├── cuda_graph_decoder_wrapper.py         # CUDA Graph 解码器加速包装
├── pipeline.yaml                         # 多阶段流水线配置
├── qwen3_tts.py                          # 统一入口模型（HF 模型 + vLLM 适配）
├── qwen3_tts_code2wav.py                 # Code2Wav 阶段（codec → 波形）
├── qwen3_tts_code_predictor_vllm.py      # Code Predictor（vLLM 版本）
├── qwen3_tts_talker.py                   # Talker 模型（AR 文本→codec）
├── qwen3_tts_tokenizer.py                # 语音分词器封装（编码/解码）
├── voice_cache_manager.py                # 声音克隆缓存管理
├── tokenizer_12hz/                       # 12Hz 语音分词器
│   ├── __init__.py
│   ├── configuration_qwen3_tts_tokenizer_v2.py
│   └── modeling_qwen3_tts_tokenizer_v2.py
└── tokenizer_25hz/                       # 25Hz 语音分词器
    ├── __init__.py
    ├── configuration_qwen3_tts_tokenizer_v1.py
    ├── modeling_qwen3_tts_tokenizer_v1.py
    └── vq/                               # 矢量量化模块
        ├── __init__.py
        ├── core_vq.py
        ├── speech_vq.py
        └── whisper_encoder.py
```

## 整体架构

Qwen3-TTS 是一个**纯文本到语音（TTS）**模型，不同于 Qwen2.5-Omni 和 Qwen3-Omni 的多模态理解+语音合成。它采用两阶段流水线：

```
文本输入（含说话人/语言/风格指令）
        │
        ▼
  ┌──────────────────┐
  │     Talker       │  阶段0: AR 生成 codec codes
  │  (Qwen3 Decoder) │  支持 Code Predictor MTP
  │  + Speaker Enc.  │  ECAPA-TDNN 说话人编码
  └──────┬───────────┘
         │ 多层 codec tokens
         ▼
  ┌──────────────────┐
  │    Code2Wav      │  阶段1: codec → 音频波形
  │  (SpeechTokenizer│  支持 CUDA Graph 加速
  │   Decoder)       │  支持流式分块解码
  └──────────────────┘
```

## 与 Qwen3-Omni 的区别

| 特性 | Qwen3-Omni | Qwen3-TTS |
|------|-----------|-----------|
| 输入 | 多模态（文本+图像+视频+音频） | 纯文本 |
| Thinker | 有（MoE Transformer） | 无 |
| 声音克隆 | 基于 speaker_id | 支持 ICL/x-vector 两种模式 |
| Code2Wav | 自定义 Conv+Transformer | SpeechTokenizer 解码器 |
| 语音分词器 | 无 | 25Hz/12Hz 两种版本 |
| CUDA Graph | 无 | 支持（Code2Wav 加速） |

## 核心设计特点

1. **声音克隆**: 支持两种模式 —— ICL（in-context learning，带参考音频 codec）和 x-vector（仅说话人嵌入）
2. **双速率分词器**: 25Hz（SpeechVQ + BigVGAN）和 12Hz（端到端编解码器），适配不同质量/速度需求
3. **CUDA Graph 加速**: 预捕获固定大小的解码器计算图，显著减少 kernel launch 开销
4. **安全缓存**: 使用 safetensors（非 pickle）存储声音克隆缓存，防止 RCE 攻击
5. **异步流式**: pipeline.yaml 配置异步分块，共享内存连接器传递 codec 流

## 数据流说明

| 阶段 | 输入 | 输出 | 关键类 |
|------|------|------|--------|
| Talker | 文本 tokens + 说话人信息 | Layer-0 codec + MTP codes | `Qwen3TTSTalkerForConditionalGeneration` |
| Code2Wav | 多层 codec tokens | 音频波形 | `Qwen3TTSCode2Wav` |
| 语音分词器 | 参考音频 | codec codes + x-vector | `Qwen3TTSTokenizer` |
| 缓存管理 | 说话人音频 | 缓存的声音克隆数据 | `VoiceCacheManager` |
