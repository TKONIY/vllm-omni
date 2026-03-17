# `mimo_audio_code2wav.py` — Code2Wav 阶段

## 文件概述

MiMo-Audio 的 token 到波形转换阶段。包含 `MiMoAudioTokenizerWorker`（管理编解码）和 `MiMoAudioToken2WavForConditionalGenerationVLLM`（vLLM 集成封装）。

## 关键代码解析

### TokenizerWorker

```python
class MiMoAudioTokenizerWorker:
    def __init__(self, device_str, config_path, audio_tokenizer_path):
        self.audio_tokenizer = MiMoAudioTokenizer.from_pretrained(audio_tokenizer_path)
        self.mel_transform = MelSpectrogram(sample_rate=24000, n_fft=1024, ...)
    def encode(self, audio):
        """wav → mel → encoder → RVQ → codes [audio_channels, T]"""
        mel = self.wav2mel(wav).transpose(0, 1)
        audio_codes = self.encode_batch_base(feature_groups, len_groups)
    def decode(self, tokens):
        """codes [audio_channels, T] → decoder → wav"""
        decoded_audio = self.audio_tokenizer.decode(tokens)
```

### 代码提取工具

```python
def extract_audio_code_tensor(flat_codes, group_size, audio_channels, codes):
    """从扁平化的 talker 输出中提取 [audio_channels, T] 格式的编码"""
    groups = flat_codes.view(-1, group_size, audio_channels + 1)
    for group in groups:
        text_token = group[0, 0]
        if text_token == codes.empty:
            audio_buffer.append(group[:, 1:])  # 去掉文本列
        elif text_token == codes.eostm:
            break
    return torch.cat(audio_buffer).transpose(0, 1)
```

### 流式分块解码

```python
class MiMoAudioToken2WavForConditionalGenerationVLLM:
    def chunked_decode_streaming(self, codes, chunk_size=10, left_context_size=10):
        """分块解码，裁剪左上下文实现无缝拼接"""
        wav_chunk = self._decode_waveform_from_codes(codes)
        drop = context_size * self.total_upsample
        return wav_chunk[drop:]
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `MiMoAudioTokenizerWorker` | 类 | 编解码管理器（带缓存） |
| `AudioStreamerConfig` | dataclass | 流式配置 |
| `MiMoAudioCodes` | dataclass | 特殊 token ID 集合 |
| `extract_audio_code_tensor()` | 函数 | 扁平编码→多通道格式 |
| `get_tokenizer_worker()` | 函数 | 缓存的 Worker 工厂 |
| `MiMoAudioToken2WavForConditionalGenerationVLLM` | 类 | vLLM 集成的 Code2Wav |

## 总结

Code2Wav 阶段的核心是 TokenizerWorker，它管理 mel 频谱计算、RVQ 编码/解码、以及波形合成。Worker 使用进程级缓存避免重复加载。
