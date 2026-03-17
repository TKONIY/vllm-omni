# `config_mimo_audio.py` — MiMo-Audio 配置类

## 文件概述

定义两个配置类：`MiMoAudioConfig`（LLM + 音频通道配置）和 `MiMoAudioTokenizerConfig`（音频编解码器配置）。还定义了关键的特殊 token ID 常量。

## 关键代码解析

### 特殊 token 常量

```python
SPAN_CODEC_START_TOKEN_ID = 151670  # 音频 span 起始
SPAN_CODEC_END_TOKEN_ID = 151672    # 音频 span 结束
TALKER_CODEC_PAD_TOKEN_ID = 151667  # codec 填充 token
TEXT_GROUP_SIZE = 5                  # 文本 token 分组大小
PAD_GROUP_SIZE = 5                   # 填充 token 分组大小
```

### MiMoAudioConfig

```python
class MiMoAudioConfig(Qwen2Config):
    """继承 Qwen2 配置，添加音频相关参数"""
    speech_vocab_size = "1025-1025-129-129-129-129-129-129"  # 8通道词表大小
    delay_pattern = "0-1-2-3-4-5-6-7"                       # 延迟模式
    audio_channels = 8                                        # 音频通道数
    group_size = 4                                            # 帧分组大小
    local_dim = 1024                                          # 局部 Transformer 维度
    local_layers = 16                                         # 局部 Transformer 层数

    def local_config(self):
        """生成局部 Transformer 的 Qwen2 子配置"""
        config.hidden_size = self.local_dim
        config.num_hidden_layers = self.local_layers
```

### MiMoAudioTokenizerConfig

```python
class MiMoAudioTokenizerConfig(PretrainedConfig):
    """音频 tokenizer 的完整配置"""
    model_type = "mimo_audio_tokenizer"
    sampling_rate = 24000
    encoder_layers = 8    # 编码器 Transformer 层
    decoder_layers = 8    # 解码器 Transformer 层
    num_quantizers = 12   # RVQ 量化器数量
    vocoder_num_layers = 30  # Vocos 解码器层数
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `MiMoAudioConfig` | 类 | LLM + 音频通道配置 |
| `MiMoAudioTokenizerConfig` | 类 | 音频编解码器配置 |
| `parsed_speech_vocab_sizes()` | 方法 | 解析多通道词表大小 |
| `parsed_delay_pattern()` | 方法 | 解析延迟模式 |
| `local_config()` | 方法 | 生成局部 Transformer 配置 |

## 总结

配置类将 MiMo-Audio 的多通道音频参数用字符串格式（如 "1025-1025-129-..."）紧凑表示，通过 `_parse_maybe_list` 方法统一解析。
