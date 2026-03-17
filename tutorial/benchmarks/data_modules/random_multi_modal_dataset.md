# `random_multi_modal_dataset.py` — 多模态随机数据集生成

## 文件概述

该文件实现了 `OmniRandomMultiModalDataset` 类和 `process_audio` 辅助函数，用于在基准测试中生成包含音频、图像和视频的合成多模态数据。它继承 vLLM 的 `RandomMultiModalDataset`，扩展了音频生成和视频生成能力。

## 关键代码解析

### 音频处理函数

```python
def process_audio(audio: Any) -> Mapping[str, Any]:
    if isinstance(audio, dict) and "bytes" in audio:
        audio_bytes = audio["bytes"]
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/mpeg;base64,{audio_base64}"},
        }
    if isinstance(audio, str):
        audio_url = audio if audio.startswith(("http://", "https://", "file://")) else f"file://{audio}"
        return {"type": "audio_url", "audio_url": {"url": audio_url}}
```

`process_audio` 将音频输入统一转换为 OpenAI API 兼容的字典格式，支持两种输入：
1. 包含原始字节的字典 `{"bytes": raw_bytes}` -> Base64 编码的 data URL
2. 字符串路径或 URL -> 直接构造 audio_url

### 合成音频生成

```python
def generate_synthetic_audio(self, duration: int, num_channels: int) -> dict[str, Any]:
    sample_rate = 48000
    num_samples = int(sample_rate * duration)
    audio_data = self._rng.uniform(-0.5, 0.5, (num_samples, num_channels))
    audio_data = np.clip(audio_data, -1.0, 1.0)
    # ... 转换为 WAV 格式字节
```

使用 48kHz 采样率生成指定时长和通道数的随机噪声音频，输出为 WAV 格式字节。

### 模态配置映射

```python
def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
    if config[0] == 0:
        return "audio"
    elif config[-1] == 1:
        return "image"
    elif config[-1] > 1:
        return "video"
```

通过三元组 `(height, width, frames)` 的约定来区分模态类型：
- `height == 0` 表示音频
- `frames == 1` 表示图像
- `frames > 1` 表示视频

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `process_audio(audio)` | 函数 | 将音频输入转换为 OpenAI API 兼容格式 |
| `OmniRandomMultiModalDataset` | 类 | 继承 `RandomMultiModalDataset`，扩展音频/视频合成能力 |
| `generate_synthetic_audio(duration, num_channels)` | 方法 | 生成随机 WAV 音频 |
| `generate_synthetic_video(width, height, num_frames)` | 方法 | 生成随机 MP4 视频 |
| `generate_mm_item(mm_item_config)` | 方法 | 根据配置生成对应模态的合成数据 |
| `map_config_to_modality(config)` | 方法 | 三元组配置到模态类型的映射 |

## 与其他模块的关系

- **继承 vLLM**: 继承 `vllm.benchmarks.datasets.RandomMultiModalDataset`，复用图像生成逻辑。
- **被 patch 模块调用**: `patch/patch.py` 中的 `get_samples` 函数实例化 `OmniRandomMultiModalDataset` 来生成测试样本。
- **依赖外部库**: 使用 `numpy`、`soundfile`（音频）、`imageio`（视频）、`torch` 进行数据生成。

## 总结

该文件是基准测试数据生成的核心，通过继承和扩展 vLLM 的随机数据集类，增加了音频和视频合成能力，使 vllm-omni 能够对多模态推理场景进行全面的性能测试。
