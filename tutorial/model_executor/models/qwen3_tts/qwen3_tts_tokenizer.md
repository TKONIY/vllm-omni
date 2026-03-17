# `qwen3_tts_tokenizer.py` — 语音分词器封装

## 文件概述

本文件实现了 `Qwen3TTSTokenizer`，一个统一封装 25Hz 和 12Hz 两种语音分词器的接口类。支持多种音频输入格式（文件路径、URL、base64、numpy 数组），提供 encode（音频→codec）和 decode（codec→音频）双向转换。

## 关键代码解析

### 1. 统一加载

```python
class Qwen3TTSTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)
        inst.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
```

通过 AutoConfig/AutoModel 注册机制自动识别 25Hz 或 12Hz 分词器。

### 2. 音频输入归一化

```python
def _normalize_audio_inputs(self, audios, sr):
    target_sr = int(self.feature_extractor.sampling_rate)
    if isinstance(audios[0], str):
        return [self.load_audio(x, target_sr=target_sr) for x in audios]
    # numpy 输入：重采样到目标采样率
    for a in audios:
        if int(sr) != target_sr:
            a = librosa.resample(y=a, orig_sr=int(sr), target_sr=target_sr)
```

### 3. 编码

```python
def encode(self, audios, sr=None, return_dict=True):
    wavs = self._normalize_audio_inputs(audios, sr=sr)
    inputs = self.feature_extractor(raw_audio=wavs, ...)
    with torch.inference_mode():
        enc = self.model.encode(inputs["input_values"].squeeze(1), inputs["padding_mask"].squeeze(1))
    return enc
```

### 4. 解码

```python
def decode(self, encoded):
    model_type = self.model.get_model_type()
    if model_type == "qwen3_tts_tokenizer_25hz":
        # 需要 audio_codes + xvectors + ref_mels
        dec = self.model.decode(audio_codes_padded, xvectors_batch, ref_mels_padded)
    elif model_type == "qwen3_tts_tokenizer_12hz":
        # 只需要 audio_codes
        dec = self.model.decode(audio_codes_padded)
    return wavs, int(self.model.get_output_sample_rate())
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3TTSTokenizer` | 类 | 统一分词器接口 |
| `from_pretrained()` | 类方法 | HF 风格加载 |
| `encode()` | 方法 | 音频 → codec codes |
| `decode()` | 方法 | Codec codes → 音频 |
| `load_audio()` | 方法 | 加载和重采样音频 |
| `get_model_type()` | 方法 | 获取底层模型类型 |

## 与其他模块的关系

- **被引用**: `qwen3_tts.py` 和 `qwen3_tts_code2wav.py` 使用
- **依赖**: `tokenizer_12hz/` 和 `tokenizer_25hz/` 子模块

## 总结

`Qwen3TTSTokenizer` 提供了一个优雅的统一接口，隐藏了 25Hz/12Hz 两种分词器的差异。25Hz 版本需要额外的 x-vector 和参考梅尔频谱进行解码（支持声音克隆），12Hz 版本则更简洁。
