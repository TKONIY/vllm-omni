# `utils.py` — CosyVoice3 工具函数集

## 文件概述

提供 CosyVoice3 所需的音频处理工具函数，包括 mel 频谱计算、音频加载/重采样、语音 token 提取、说话人嵌入提取等。

## 关键代码解析

### mel 频谱计算

```python
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """计算 mel 频谱图，使用 librosa mel 滤波器组"""
    mel = _get_mel_basis(...)          # LRU 缓存的 mel 基
    window = _get_hann_window(...)     # LRU 缓存的 Hann 窗
    spec = torch.stft(y, n_fft, ...)   # STFT
    spec = torch.matmul(mel, spec)     # mel 投影
    spec = spectral_normalize_torch(spec)  # 动态范围压缩
```

### 音频特征提取

```python
def extract_speech_token(prompt_wav, speech_tokenizer_session, device):
    """使用 ONNX 语音 tokenizer 提取语音 token"""
    speech = load_wav(prompt_wav, 16000)       # 重采样到 16kHz
    feat = log_mel_spectrogram(speech, n_mels=128)  # Whisper 风格 mel
    speech_token = speech_tokenizer_session.run(None, {...})  # ONNX 推理

def extract_spk_embedding(prompt_wav, campplus_session, device):
    """使用 CamPPlus 提取说话人嵌入"""
    feat = kaldi.fbank(speech, num_mel_bins=80)  # FBank 特征
    embedding = campplus_session.run(None, {...})  # ONNX 推理
```

### pad mask 工具

```python
def make_pad_mask(lengths, max_len=0):
    """生成填充掩码，lengths 指定每个序列的有效长度"""
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `mel_spectrogram()` | 函数 | 计算 mel 频谱图 |
| `log_mel_spectrogram()` | 函数 | Whisper 风格 log-mel（16kHz, 128 bins）|
| `load_wav()` | 函数 | 加载并重采样音频 |
| `extract_speech_feat()` | 函数 | 提取 24kHz mel 特征 |
| `extract_speech_token()` | 函数 | ONNX 语音 tokenizer 推理 |
| `extract_spk_embedding()` | 函数 | CamPPlus 说话人嵌入提取 |
| `extract_text_token()` | 函数 | 文本 tokenizer 编码 |
| `concat_text_with_prompt_ids()` | 函数 | 拼接提示文本与目标文本 |
| `make_pad_mask()` | 函数 | 生成批次填充掩码 |

## 总结

工具函数集覆盖了 TTS 管线所需的所有音频预处理步骤，使用 LRU 缓存优化频繁调用的 mel 基和窗函数。
