# `tokenizer_12hz/` — 12Hz 语音分词器

## 目录结构

```
tokenizer_12hz/
├── __init__.py                                    # 空初始化文件
├── configuration_qwen3_tts_tokenizer_v2.py        # V2 分词器配置
└── modeling_qwen3_tts_tokenizer_v2.py             # V2 分词器模型
```

## 文件概述

12Hz 语音分词器（V2）是一个端到端的编解码器，以 12Hz 的帧率对音频进行编码/解码。相比 25Hz 版本，12Hz 版本更简洁：编码仅产出 `audio_codes`，解码时不需要额外的 x-vector 或参考梅尔频谱。

### `configuration_qwen3_tts_tokenizer_v2.py`

定义 `Qwen3TTSTokenizerV2Config`，包含编码器和解码器的超参数（隐藏维度、层数、码本大小等）。模型类型为 `"qwen3_tts_tokenizer_12hz"`。

### `modeling_qwen3_tts_tokenizer_v2.py`

实现 `Qwen3TTSTokenizerV2Model`，包含：
- **编码器**: 音频波形 → 离散 codes
- **解码器**: 离散 codes → 音频波形
- **`Qwen3TTSTokenizerV2EncoderOutput`**: 编码输出数据类（`audio_codes`）

## 核心类/函数

| 名称 | 文件 | 说明 |
|------|------|------|
| `Qwen3TTSTokenizerV2Config` | configuration | 配置类 |
| `Qwen3TTSTokenizerV2Model` | modeling | 编解码器模型 |
| `Qwen3TTSTokenizerV2EncoderOutput` | modeling | 编码输出 |

## 与其他模块的关系

- **被引用**: `qwen3_tts_tokenizer.py` 通过 AutoModel 加载
- **区别**: 12Hz 更低帧率，解码更简单（无需条件信息）

## 总结

12Hz 分词器是 Qwen3-TTS 的轻量级音频编解码方案，适用于对延迟敏感但对音质要求稍低的场景。
