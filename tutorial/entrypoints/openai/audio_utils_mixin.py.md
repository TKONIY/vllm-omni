# `audio_utils_mixin.py` — 音频工具 Mixin

## 文件概述

提供 `AudioMixin` 类，封装了音频张量到字节的转换逻辑，支持多种音频格式和速度调节。被 `OmniOpenAIServingChat` 和 `OmniOpenAIServingSpeech` 混入使用。

## 关键代码解析

### 音频格式转换

```python
class AudioMixin:
    def create_audio(self, audio_obj: CreateAudio) -> AudioResponse:
        """将音频张量转为指定格式的字节"""
        supported_formats = {
            "wav":  ("WAV",  "audio/wav",  {}),
            "pcm":  ("RAW",  "audio/pcm",  {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3":  ("MP3",  "audio/mpeg", {}),
            "aac":  ("AAC",  "audio/aac",  {}),
            "opus": ("OGG",  "audio/ogg",  {"subtype": "OPUS"}),
        }

        # 应用速度调节
        audio_tensor, sample_rate = self._apply_speed_adjustment(
            audio_tensor, speed, sample_rate
        )

        # 使用 soundfile 编码
        with BytesIO() as buffer:
            soundfile.write(buffer, audio_tensor, sample_rate,
                          format=soundfile_format, **kwargs)
            audio_data = buffer.getvalue()

        # 可选 base64 编码
        if base64_encode:
            audio_data = base64.b64encode(audio_data).decode("utf-8")
        return AudioResponse(audio_data=audio_data, media_type=media_type)
```

### 速度调节

```python
def _apply_speed_adjustment(self, audio_tensor, speed, sample_rate):
    """使用 librosa 的时间拉伸实现变速不变调"""
    if speed == 1.0:
        return audio_tensor, sample_rate
    stretched_audio = librosa.effects.time_stretch(y=audio_tensor, rate=speed)
    return stretched_audio, sample_rate
```

使用 `librosa.effects.time_stretch` 实现变速不变调，保持音高不变。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `AudioMixin` | Mixin 类 | 音频格式转换和速度调节 |
| `create_audio()` | 方法 | 将音频张量转为字节 |
| `_apply_speed_adjustment()` | 方法 | 变速不变调 |

## 与其他模块的关系

- 被 `serving_chat.py` 的 `OmniOpenAIServingChat` 混入
- 被 `serving_speech.py` 的 `OmniOpenAIServingSpeech` 混入
- 使用 `protocol/audio.py` 的 `CreateAudio` 和 `AudioResponse` 数据模型

## 总结

一个专注于音频编码的 Mixin，支持 6 种音频格式和变速功能，通过混入模式为多个服务类提供音频处理能力。
