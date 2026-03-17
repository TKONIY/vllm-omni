# `video.py` — 视频音频提取工具

## 文件概述

`video.py` 提供从视频文件中提取音频信号的工具函数，主要用于多模态模型的音频预处理。使用 `librosa` 库进行音频加载和重采样。

## 关键代码解析

```python
def extract_video_audio(path: str = None, sampling_rate: int = 16000) -> np.ndarray:
    if not path:
        path = VideoAsset(name="baby_reading").video_path
    audio_signal, sr = librosa.load(path, sr=sampling_rate)
    return audio_signal
```

工作流程：
1. 如果未提供路径，使用 vLLM 内置的 `baby_reading` 测试视频
2. 通过 `librosa.load` 加载音频，自动重采样到目标采样率（默认 16kHz）
3. 返回 numpy 数组形式的音频波形

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `extract_video_audio` | 函数 | 从视频文件提取音频波形 |

## 与其他模块的关系

- 依赖 `vllm.assets.video.VideoAsset` 获取默认测试视频
- 依赖 `librosa` 进行音频处理
- 可被多模态输入预处理模块调用

## 总结

轻量级的音频提取工具，封装了 librosa 的基本音频加载功能，为多模态推理提供标准化的音频输入。
