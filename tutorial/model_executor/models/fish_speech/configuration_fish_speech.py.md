# `configuration_fish_speech.py` — Fish Speech 配置类

## 文件概述

定义 Fish Speech S2 Pro 的三层配置体系，将 Fish Speech 原始字段名（dim、n_head、n_layer 等）映射为 Transformers/Qwen3 标准属性名，使 vLLM 的 Qwen3Model 可以直接使用。

## 关键代码解析

### Slow AR 配置映射

```python
class FishSpeechSlowARConfig(PretrainedConfig):
    model_type = "fish_qwen3"
    def __init__(self, dim=2560, n_head=32, n_local_heads=8, ...):
        self.hidden_size = dim                    # dim → hidden_size
        self.num_attention_heads = n_head         # n_head → num_attention_heads
        self.num_key_value_heads = n_local_heads  # n_local_heads → num_key_value_heads
        self.num_hidden_layers = n_layer          # n_layer → num_hidden_layers
        self.codebook_size = codebook_size        # 4096
        self.num_codebooks = num_codebooks        # 10
```

### 顶层配置

```python
class FishSpeechConfig(PretrainedConfig):
    model_type = "fish_qwen3_omni"
    sub_configs = {
        "text_config": FishSpeechSlowARConfig,
        "audio_decoder_config": FishSpeechFastARConfig,
    }
    def get_text_config(self):
        return self.text_config  # vLLM 通过此方法获取 LLM 配置
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `FishSpeechSlowARConfig` | 类 | Slow AR (Qwen3) 配置 |
| `FishSpeechFastARConfig` | 类 | Fast AR (4层) 配置 |
| `FishSpeechConfig` | 类 | 顶层配置，包装两个子配置 |

## 总结

配置类的核心价值在于字段名映射，使 Fish Speech 的非标准字段能被 vLLM 的标准 Qwen3 实现直接消费。
