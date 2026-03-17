# `dac_encoder.py` — DAC 编码器（语音克隆用）

## 文件概述

实现参考音频的 DAC 编码，用于 Fish Speech 的语音克隆功能。在 API 服务器进程中以 CPU 运行，首次使用时懒加载 codec.pth。

## 关键代码解析

```python
@torch.no_grad()
def encode_reference_audio(model_path, wav_samples, sample_rate) -> list[int]:
    """将参考音频编码为语义 token ID 列表"""
    codec = _load_dac_codec(model_path)            # 懒加载 + 缓存
    wav = _resample(wav, sample_rate, DAC_SAMPLE_RATE)  # 重采样到 44100Hz
    codes, _ = codec.encode(wav_tensor, feature_lengths)  # VQ 编码
    semantic_codes = codes[0, 0, :].tolist()        # 取第 0 个 codebook
    # 转换为 token ID: 151678 + code_value
    semantic_token_ids = [151678 + int(c) for c in semantic_codes]
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `_load_dac_codec()` | 函数 | 懒加载 DAC codec（带缓存） |
| `_resample()` | 函数 | torchaudio 重采样 |
| `encode_reference_audio()` | 函数 | 参考音频 → 语义 token ID 列表 |

## 总结

DAC 编码器是语音克隆的入口，将参考音频的语义特征编码为可注入 Slow AR prompt 的 token 序列。
