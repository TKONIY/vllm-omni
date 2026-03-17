# `serving_speech.py` — 语音合成服务

## 文件概述

`OmniOpenAIServingSpeech` 实现了 OpenAI 兼容的 TTS (Text-to-Speech) API，提供 `/v1/audio/speech` 端点。支持多种 TTS 模型（如 Qwen3-TTS、FishSpeech），提供声音克隆（voice cloning）、说话人管理、流式 PCM 输出等高级功能。

## 关键代码解析

### TTS 配置

```python
_TTS_MODEL_STAGES: set[str] = {"qwen3_tts", "fish_speech_slow_ar"}
_TTS_LANGUAGES: set[str] = {"Auto", "Chinese", "English", "Japanese", ...}
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
```

### 流式 WAV 头

```python
def _create_wav_header(sample_rate, num_channels=1, bits_per_sample=16):
    """创建流式 WAV 头，使用占位符大小"""
    # 使用 0xFFFFFFFF 作为大小占位符，兼容 OpenAI 的流式 WAV 实现
    header = struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", placeholder_size, b"WAVE", ...)
```

### 说话人管理

服务提供了完整的说话人 CRUD API：

```python
# 上传说话人音频样本
async def upload_speaker(self, speaker_name, audio_file, ...)
# 获取说话人列表
async def list_speakers(self)
# 删除说话人
async def delete_speaker(self, speaker_name)
```

上传的说话人音频被预处理（重采样到 16kHz）并生成参考码缓存，后续请求可以直接使用缓存的说话人特征。

### 流式 PCM 生成

```python
async def _generate_pcm_chunks(self, generator, request_id):
    """逐块产出 PCM 音频数据"""
    async for output in generator:
        audio_tensor = output.request_output.audio
        pcm_bytes = (audio_tensor * 32767).to(torch.int16).numpy().tobytes()
        yield pcm_bytes
```

流式模式下，TTS 模型每解码一个 chunk 就立即将 PCM 数据发送给客户端，实现低延迟的音频输出。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniOpenAIServingSpeech` | 类 | TTS 服务处理器 |
| `create_speech()` | 异步方法 | 处理语音合成请求 |
| `upload_speaker()` | 异步方法 | 上传说话人音频 |
| `list_speakers()` | 异步方法 | 列出说话人 |
| `delete_speaker()` | 异步方法 | 删除说话人 |
| `_generate_pcm_chunks()` | 异步生成器 | 流式 PCM 音频生成 |
| `_create_wav_header()` | 函数 | 创建流式 WAV 头 |
| `_sanitize_filename()` | 函数 | 文件名安全处理 |

## 与其他模块的关系

- 混入 `AudioMixin`（`audio_utils_mixin.py`）获取音频编码能力
- 使用 `MetadataManager`（`metadata_manager.py`）管理说话人元数据
- 使用 `AsyncOmni` 引擎驱动 TTS 管线
- 使用 `protocol/audio.py` 的请求/响应数据模型
- 被 `serving_speech_stream.py` 复用内部方法

## 总结

`OmniOpenAIServingSpeech` 是一个功能丰富的 TTS 服务层，支持多种 TTS 模型、声音克隆、说话人管理和流式音频输出。它将复杂的 TTS 管线封装为简洁的 OpenAI 兼容 API，是 vLLM-Omni 在语音合成领域的核心组件。
