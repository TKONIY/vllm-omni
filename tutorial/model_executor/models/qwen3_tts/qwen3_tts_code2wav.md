# `qwen3_tts_code2wav.py` — Code2Wav 阶段

## 文件概述

本文件实现了 Qwen3-TTS 的 Code2Wav 模型，使用 SpeechTokenizer 解码器将多层 RVQ codec tokens 转换为音频波形。与 Qwen3-Omni 的 Code2Wav 不同，此版本直接使用预训练的语音分词器解码器，而非自定义的 Conv+Transformer 架构。

## 关键代码解析

### 1. 延迟加载

```python
class Qwen3TTSCode2Wav(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self._speech_tokenizer: Qwen3TTSTokenizer | None = None
        self._decoder: nn.Module | None = None
        self._num_quantizers: int | None = None

    def _ensure_speech_tokenizer_loaded(self):
        cfg_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        tok = Qwen3TTSTokenizer.from_pretrained(speech_tokenizer_dir, ...)
        tok.model.to(device=self.vllm_config.device_config.device)
```

解码器在首次使用时才加载，避免不必要的初始化开销。

### 2. 模型属性

```python
self.have_multimodal_outputs = True        # 产生多模态输出
self.enable_update_additional_information = True  # 支持运行时信息更新
self.requires_raw_input_tokens = True      # 需要原始 token（非嵌入）
```

### 3. 流式解码支持

Code2Wav 通过 `left_context_size` 参数支持流式解码，每个请求可以有不同的上下文大小。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3TTSCode2Wav` | 类 | Code2Wav 主模型 |
| `_ensure_speech_tokenizer_loaded()` | 方法 | 延迟加载语音分词器 |
| `_module_device()` | 静态方法 | 获取模块设备 |

## 与其他模块的关系

- **被引用**: `pipeline.yaml` 中作为 Stage 1 的模型架构
- **依赖**: `qwen3_tts_tokenizer.py` 中的 `Qwen3TTSTokenizer`
- **上游**: 接收 Talker 生成的多层 codec tokens

## 总结

Code2Wav 是 Qwen3-TTS 流水线的最后阶段。它封装了预训练的 SpeechTokenizer 解码器，支持 25Hz 和 12Hz 两种分词器版本。关键特性包括延迟加载（减少初始化时间）和流式解码支持。
