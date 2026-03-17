# `modeling_audio_tokenizer.py` — MiMo-Audio Tokenizer 模型

## 文件概述

实现完整的 MiMo-Audio 音频 tokenizer，包含编码器（AudioEncoder）、解码器（AudioDecoder）和声码器（TransformerVocos）。约 825 行，是一个自包含的音频编解码系统。

## 关键代码解析

### ISTFT 合成头

```python
class ISTFTHead(nn.Module):
    """ISTFT 波形合成：Linear → 分离幅度/相位 → ISTFT"""
    def forward(self, x):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        S = mag.float() * (torch.cos(p).float() + 1j * torch.sin(p).float())
        return self.istft(S)
```

### Transformer Vocos 声码器

```python
class TransformerVocos(nn.Module):
    """基于 Transformer + ISTFT 的声码器"""
    def __init__(self, config):
        self.layers = nn.ModuleList([TransformerLayer(...) for _ in range(30)])
        self.head = ISTFTHead(dim, n_fft=1024, hop_length=240)
    def forward(self, x, input_length):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, input_length, rope_embeddings)
        return self.head(x)  # → waveform
```

### 音频编码器

```python
class AudioEncoder(nn.Module):
    """Conv前端 + Transformer + AvgPool下采样 + RVQ量化"""
    def encode(self, input_features, input_lens):
        hidden = self.get_features(input_features, output_length)
        codes = self.quantizer.encode(hidden.float())  # RVQ 编码
        return codes, output_length
```

### 音频解码器

```python
class AudioDecoder(nn.Module):
    """CausalConvTranspose上采样 + Transformer + Vocos声码器"""
    def forward(self, audio_embed, input_length):
        audio_embed, length = self.dconv1(audio_embed, input_length)  # 上采样
        for layer in self.layers:
            hidden = layer(hidden, length, rope_embeddings)
        coarse_mel, length = self.dconv2(hidden, length)  # 生成粗糙mel
        return self.vocoder(coarse_mel, length)  # mel → wav
```

### MiMoAudioTokenizer

```python
class MiMoAudioTokenizer(PreTrainedModel):
    config_class = MiMoAudioTokenizerConfig
    def __init__(self, config):
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)
    def decode(self, codes):
        hidden = self.encoder.decode_vq(codes)  # RVQ 解码
        return self.decoder(hidden, torch.tensor([hidden.size(0)]))
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `ISTFT` | 类 | 自定义 ISTFT（支持 "same" padding） |
| `ISTFTHead` | 类 | ISTFT 波形合成头 |
| `RotaryEmbedding` | 类 | RoPE 位置编码 |
| `Attention` | 类 | 多头注意力（Flash/SDPA） |
| `TransformerLayer` | 类 | Transformer 层 |
| `TransformerVocos` | 类 | Vocos 声码器 |
| `AudioEncoder` | 类 | 音频编码器 |
| `AudioDecoder` | 类 | 音频解码器 |
| `CausalConvTranspose1d` | 类 | 因果转置卷积 |
| `MiMoAudioTokenizer` | 类 | 完整 tokenizer |

## 总结

MiMo-Audio Tokenizer 是一个完整的神经音频编解码器，核心路径为：mel → Conv前端 → Transformer → AvgPool → RVQ 量化 → RVQ 解码 → ConvTranspose → Transformer → Vocos(ISTFT) → 波形。支持流式解码。
