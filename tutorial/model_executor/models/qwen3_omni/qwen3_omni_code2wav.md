# `qwen3_omni_code2wav.py` — Code2Wav 模型

## 文件概述

本文件实现了 Qwen3-Omni MoE 的 Code2Wav 模型，将 16 层 RVQ codec codes 转换为音频波形。采用 Embedding + Transformer + 上采样 + 解码器的四阶段架构，总上采样因子约 1280x（100 个 codec 帧 → 128000 个音频样本，约 8 秒 @16kHz）。

## 关键代码解析

### 1. 模型架构

```python
class Qwen3OmniMoeCode2Wav(nn.Module):
    def __init__(self, ...):
        # 阶段1: 代码嵌入（所有 RVQ 层共享一个嵌入表）
        self.code_embedding = nn.Embedding(
            config.codebook_size * config.num_quantizers, config.hidden_size)
        # 偏移量：层 0: 0~1023, 层 1: 1024~2047, ...
        self.register_buffer("code_offset",
            torch.arange(config.num_quantizers).view(1,-1,1) * config.codebook_size)

        # 阶段2: 预 Transformer（添加时间上下文）
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(...)

        # 阶段3: 上采样（ConvNeXt + 转置卷积）
        self.upsample = nn.ModuleList([...])

        # 阶段4: 解码器（渐进式上采样 → 波形）
        self.decoder = nn.ModuleList([...SnakeBeta...])
```

### 2. 前向传播

```python
def forward(self, codes: torch.Tensor) -> torch.Tensor:
    # [batch, num_quantizers, seq_len]
    hidden = self.code_embedding(codes + self.code_offset).mean(1)  # 平均所有 RVQ 层
    hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
    hidden = hidden.permute(0, 2, 1)  # → [batch, hidden, seq]
    for blocks in self.upsample:
        for block in blocks: hidden = block(hidden)
    wav = hidden
    for block in self.decoder: wav = block(wav)
    return wav.clamp(min=-1.0, max=1.0)
```

### 3. 分块解码

```python
def chunked_decode(self, codes, chunk_size=300, left_context_size=25, ...):
    while start_index < codes.shape[-1]:
        codes_chunk = codes[..., start_index - context_size : end_index]
        wav_chunk = self(codes_chunk)
        wavs.append(wav_chunk[..., context_size * self.total_upsample :])
```

使用重叠上下文避免块间不连续伪影。

### 4. 流式解码

```python
def chunked_decode_streaming(self, codes, left_context_size, seq_token_counts):
    batch_wav = self(codes)
    for idx, code_seq_len in enumerate(code_seq_lens):
        wav_chunk = batch_wav[idx, :,
            left_context_size[idx] * self.total_upsample : code_seq_len * self.total_upsample]
```

流式版本：每个请求可有不同的左上下文大小。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3OmniMoeCode2Wav` | 类 | Code2Wav 主模型 |
| `forward()` | 方法 | 完整前向传播 |
| `chunked_decode()` | 方法 | 分块解码（避免 OOM） |
| `chunked_decode_streaming()` | 方法 | 流式分块解码 |
| `load_weights()` | 方法 | HF 权重加载 |

## 与其他模块的关系

- **被引用**: `qwen3_omni.py` 通过架构名 `"Qwen3OmniMoeCode2Wav"` 实例化
- **依赖**: HuggingFace `Qwen3OmniMoeCausalConvNet`、`Qwen3OmniMoeConvNeXtBlock` 等组件
- **上游**: 接收 Talker + Code Predictor 生成的 16 层 RVQ codes

## 总结

Code2Wav 模型将离散的多层 RVQ 编码转换为连续的音频波形。核心设计包括：(1) 共享嵌入表 + 偏移量区分不同 RVQ 层；(2) 平均池化合并多层信息；(3) Transformer 添加时间上下文；(4) 渐进式上采样（~1280x）；(5) 支持分块和流式解码以处理长音频。
