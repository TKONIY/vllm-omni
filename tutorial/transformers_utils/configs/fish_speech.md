# `fish_speech.py` — Fish Speech 配置注册

## 文件概述

该文件将 Fish Speech S2 Pro 模型的配置类注册到 HuggingFace Transformers 的 `AutoConfig` 系统中，使 `AutoConfig.from_pretrained()` 能够自动识别并加载 Fish Speech 模型配置。

## 关键代码解析

```python
from transformers import AutoConfig
from vllm_omni.model_executor.models.fish_speech.configuration_fish_speech import (
    FishSpeechConfig,
    FishSpeechFastARConfig,
    FishSpeechSlowARConfig,
)

AutoConfig.register("fish_qwen3_omni", FishSpeechConfig)
AutoConfig.register("fish_qwen3", FishSpeechSlowARConfig)
AutoConfig.register("fish_qwen3_audio_decoder", FishSpeechFastARConfig)
```

注册了三个 model_type 到配置类的映射：

| model_type | 配置类 | 说明 |
|------------|--------|------|
| `fish_qwen3_omni` | `FishSpeechConfig` | 顶层组合配置 |
| `fish_qwen3` | `FishSpeechSlowARConfig` | 慢速自回归配置 |
| `fish_qwen3_audio_decoder` | `FishSpeechFastARConfig` | 快速音频解码器配置 |

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FishSpeechConfig` | 类（导入） | Fish Speech 顶层配置 |
| `FishSpeechSlowARConfig` | 类（导入） | 慢速 AR 子模型配置 |
| `FishSpeechFastARConfig` | 类（导入） | 快速 AR 音频解码器配置 |

## 与其他模块的关系

- **配置定义来源**: 实际配置类定义在 `vllm_omni.model_executor.models.fish_speech.configuration_fish_speech`。
- **被 configs/__init__.py 导入**: 导入时触发 `AutoConfig.register()` 副作用。
- **服务 Fish Speech 模型**: 使 vLLM 能够通过标准接口加载 Fish Speech 模型。

## 总结

纯注册文件，将 Fish Speech 模型的三个配置类注册到 `AutoConfig`，建立 `model_type` 字符串到配置类的映射关系。配置类的实际定义在 model_executor 模块中。
