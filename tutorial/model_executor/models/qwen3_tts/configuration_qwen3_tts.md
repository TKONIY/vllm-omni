# `configuration_qwen3_tts.py` — 模型配置类

## 文件概述

本文件定义了 Qwen3-TTS 的三层嵌套配置体系：`Qwen3TTSConfig`（顶层）→ `Qwen3TTSTalkerConfig`（Talker）→ `Qwen3TTSTalkerCodePredictorConfig`（Code Predictor），以及独立的 `Qwen3TTSSpeakerEncoderConfig`（说话人编码器）。

## 关键代码解析

### 1. 说话人编码器配置

```python
class Qwen3TTSSpeakerEncoderConfig(PretrainedConfig):
    def __init__(self, mel_dim=128, enc_dim=1024, enc_channels=[512,512,512,512,1536],
                 enc_kernel_sizes=[5,3,3,3,1], enc_dilations=[1,2,3,4,1], ...):
```

基于 ECAPA-TDNN 架构的说话人编码器，从梅尔频谱提取说话人嵌入向量。

### 2. Code Predictor 配置

```python
class Qwen3TTSTalkerCodePredictorConfig(PretrainedConfig):
    model_type = "qwen3_tts_talker_code_predictor"
    def __init__(self, vocab_size=2048, hidden_size=1024, num_hidden_layers=5,
                 num_attention_heads=16, num_key_value_heads=8, num_code_groups=32, ...):
```

轻量级 Transformer，5 层，1024 维隐藏层，32 个 code groups（RVQ 层数）。

### 3. Talker 配置

```python
class Qwen3TTSTalkerConfig(PretrainedConfig):
    model_type = "qwen3_tts_talker"
    sub_configs = {"code_predictor_config": Qwen3TTSTalkerCodePredictorConfig}
    def __init__(self, code_predictor_config=None, vocab_size=3072, hidden_size=1024,
                 num_hidden_layers=20, codec_eos_token_id=4198, spk_id=None, ...):
```

包含特殊 token ID（codec BOS/EOS/PAD、思考模式 token、语言 ID）和说话人 ID 映射。

### 4. 顶层配置

```python
class Qwen3TTSConfig(PretrainedConfig):
    model_type = "qwen3_tts"
    sub_configs = {"talker_config": Qwen3TTSTalkerConfig, "speaker_encoder_config": Qwen3TTSSpeakerEncoderConfig}
    def __init__(self, ..., tokenizer_type=None, tts_model_size=None):
        # 伪视觉 token ID（-1），确保 MRoPE 扫描不会误匹配
        self.image_token_id = -1
        self.video_token_id = -1
```

注意 `get_text_config()` 方法在检测到 Code2Wav 架构时会删除 `rope_parameters`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3TTSSpeakerEncoderConfig` | 类 | ECAPA-TDNN 说话人编码器配置 |
| `Qwen3TTSTalkerCodePredictorConfig` | 类 | Code Predictor 配置 |
| `Qwen3TTSTalkerConfig` | 类 | Talker 配置 |
| `Qwen3TTSConfig` | 类 | 顶层配置 |
| `codec_frame_rate_hz` | 属性 | Codec 帧率（从 position_id_per_seconds 计算） |
| `get_text_config()` | 方法 | 返回文本模型配置（Code2Wav 时去除 RoPE） |

## 与其他模块的关系

- **被引用**: 所有 qwen3_tts 模块都依赖这些配置类
- **注册**: 配置类通过 `model_type` 注册到 HuggingFace AutoConfig

## 总结

配置类定义了 Qwen3-TTS 的完整超参数体系。关键设计包括：嵌套的 sub_configs 实现层级配置、伪视觉 token ID 避免 MRoPE 误匹配、以及 Code2Wav 阶段自动去除 RoPE 参数。
