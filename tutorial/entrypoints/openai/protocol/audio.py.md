# `audio.py` — 音频/TTS 协议

## 文件概述

定义了语音合成（TTS）API 相关的所有 Pydantic 数据模型，包括 REST API 请求、WebSocket 会话配置、内部音频创建参数和音频响应。

## 关键代码解析

### TTS 请求模型

```python
class OpenAICreateSpeechRequest(BaseModel):
    input: str                    # 要合成的文本
    model: str | None = None      # 模型名称
    voice: str | None = None      # 声音名称
    instructions: str | None = None  # 语音风格/情感指令
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = False          # 流式 PCM 输出

    # Qwen3-TTS 特有参数
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None   # 声音克隆参考音频
    ref_text: str | None = None    # 参考音频的文本
    x_vector_only_mode: bool | None = None  # 仅使用说话人嵌入
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = None  # 首块大小
```

### 流式约束验证

```python
@model_validator(mode="after")
def validate_streaming_constraints(self):
    if self.stream:
        if self.response_format not in ("pcm", "wav"):
            raise ValueError("流式模式仅支持 pcm/wav 格式")
        if self.speed != 1.0:
            raise ValueError("流式模式不支持变速")
    return self
```

### WebSocket 会话配置

```python
class StreamingSpeechSessionConfig(BaseModel):
    """WebSocket TTS 的首条配置消息"""
    model: str | None = None
    voice: str | None = None
    split_granularity: Literal["sentence", "clause"] = "sentence"
    stream_audio: bool = False  # WebSocket 内部的流式 PCM
    # ... 其他 TTS 参数
```

### 内部数据模型

```python
class CreateAudio(BaseModel):
    """内部音频创建参数（非 API 暴露）"""
    audio_tensor: np.ndarray   # 需要 arbitrary_types_allowed
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0

class AudioResponse(BaseModel):
    audio_data: bytes | str    # 原始字节或 Base64
    media_type: str            # MIME 类型
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OpenAICreateSpeechRequest` | Pydantic 模型 | TTS REST API 请求 |
| `StreamingSpeechSessionConfig` | Pydantic 模型 | WebSocket TTS 会话配置 |
| `CreateAudio` | Pydantic 模型 | 内部音频创建参数 |
| `AudioResponse` | Pydantic 模型 | 音频响应数据 |

## 与其他模块的关系

- `OpenAICreateSpeechRequest` 被 `api_server.py` 和 `serving_speech.py` 使用
- `StreamingSpeechSessionConfig` 被 `serving_speech_stream.py` 使用
- `CreateAudio` 和 `AudioResponse` 被 `audio_utils_mixin.py` 使用

## 总结

该文件定义了 TTS 服务的完整协议栈，覆盖了 REST API 和 WebSocket 两种接入方式，以及 Qwen3-TTS 声音克隆等高级特性的参数模型。通过 Pydantic 验证器确保流式模式下的参数约束一致性。
