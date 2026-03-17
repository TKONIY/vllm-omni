# `chat_utils.py` — 聊天工具函数

## 文件概述

该文件提供了视频音频提取的异步工具函数，主要用于从视频 URL 中提取音频数据，供多模态理解模型使用（如 Qwen2.5-Omni 的音频-视频联合理解）。

## 关键代码解析

```python
async def extract_audio_from_video_async(video_url: str) -> tuple[np.ndarray, int | float]:
    """从视频 URL 中提取音频，返回 (audio_array, sample_rate)"""
    parsed_url = urlparse(video_url)

    # 支持多种 URL 方案
    if parsed_url.scheme in ("http", "https"):
        video_data = await asyncio.to_thread(_download_video_sync, video_url)
        temp_video_file_path = await asyncio.to_thread(_write_temp_file_sync, video_data, ".mp4")
    elif parsed_url.scheme == "file":
        temp_video_file_path = url2pathname(parsed_url.path)
    elif parsed_url.scheme == "data":
        # 解码 base64 data URL
        ...

    # 使用 librosa 以 16kHz 采样率加载音频
    audio_array, sample_rate = await asyncio.to_thread(_load_audio_sync, temp_video_file_path)
    return audio_array, sample_rate
```

关键设计要点：
1. 所有阻塞 I/O 操作（下载、文件写入、音频解码）均通过 `asyncio.to_thread` 在线程池中执行
2. 支持 HTTP/HTTPS URL、本地文件路径、`file://` 协议和 `data:` Base64 URL
3. 统一以 16kHz 采样率输出，兼容语音模型的输入要求
4. 使用临时文件并在 `finally` 块中清理

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `extract_audio_from_video_async()` | 异步函数 | 从视频 URL 异步提取音频数据 |

## 与其他模块的关系

- 被 `openai/serving_chat.py` 中的 `_inject_audio_from_video_urls()` 调用
- 当 `use_audio_in_video=True` 时，需要在预处理阶段为视频消息注入对应的音频数据

## 总结

一个专注于视频音频提取的工具模块，通过线程池桥接将阻塞的多媒体 I/O 操作异步化，为多模态模型的音视频联合理解提供数据预处理支持。
