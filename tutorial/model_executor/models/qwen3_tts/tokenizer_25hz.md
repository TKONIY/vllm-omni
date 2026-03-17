# `tokenizer_25hz/` — 25Hz 语音分词器

## 目录结构

```
tokenizer_25hz/
├── __init__.py                                    # 空初始化文件
├── configuration_qwen3_tts_tokenizer_v1.py        # V1 分词器配置
├── modeling_qwen3_tts_tokenizer_v1.py             # V1 分词器模型
└── vq/                                            # 矢量量化子模块
    ├── __init__.py
    ├── core_vq.py                                 # 核心 VQ 算法
    ├── speech_vq.py                               # 语音 VQ 模型
    └── whisper_encoder.py                         # Whisper 编码器适配
```

## 文件概述

25Hz 语音分词器（V1）以 25Hz 帧率对音频进行编码，编码结果包含 `audio_codes`、`xvectors`（说话人嵌入）和 `ref_mels`（参考梅尔频谱）。解码时需要这三者才能重建音频——这使得 25Hz 版本天然支持声音克隆。

### `configuration_qwen3_tts_tokenizer_v1.py`

定义 `Qwen3TTSTokenizerV1Config`，包含编码器/解码器配置以及 VQ 相关参数。模型类型为 `"qwen3_tts_tokenizer_25hz"`。

### `modeling_qwen3_tts_tokenizer_v1.py`

实现 `Qwen3TTSTokenizerV1Model`，核心方法：
- `encode(input_values, padding_mask)` → `Qwen3TTSTokenizerV1EncoderOutput`（含 `audio_codes`, `xvectors`, `ref_mels`）
- `decode(audio_codes, xvectors, ref_mels)` → 音频波形

### `vq/core_vq.py`

实现核心矢量量化算法：
- **码本管理**: 可学习的嵌入码本
- **量化**: 最近邻查找 + 残差量化（RVQ）
- **EMA 更新**: 指数移动平均码本更新

### `vq/speech_vq.py`

将 VQ 模块整合为完整的语音 VQ 管道，包含编码器、解码器和 RVQ 量化器。

### `vq/whisper_encoder.py`

适配 Whisper 编码器作为语音特征提取前端，从音频中提取语义特征用于 VQ 编码。

## 核心类/函数

| 名称 | 文件 | 说明 |
|------|------|------|
| `Qwen3TTSTokenizerV1Config` | configuration | 配置类 |
| `Qwen3TTSTokenizerV1Model` | modeling | 编解码器模型 |
| `Qwen3TTSTokenizerV1EncoderOutput` | modeling | 编码输出（含 xvectors/ref_mels） |
| `VectorQuantize` | core_vq | 矢量量化 |
| `ResidualVQ` | core_vq | 残差矢量量化 |
| `SpeechVQ` | speech_vq | 语音 VQ 管道 |
| `WhisperEncoder` | whisper_encoder | Whisper 特征提取 |

## 与其他模块的关系

- **被引用**: `qwen3_tts_tokenizer.py` 通过 AutoModel 加载
- **被引用**: `qwen3_tts_code2wav.py` 使用解码器部分
- **区别**: 25Hz 更高帧率、更好音质、支持声音克隆

## 总结

25Hz 分词器是 Qwen3-TTS 的高质量音频编解码方案。它使用 Whisper 编码器提取语义特征，通过 RVQ 进行离散化，并在解码时利用 x-vector 和参考梅尔频谱实现高保真声音重建和声音克隆。
