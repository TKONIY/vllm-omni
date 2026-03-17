# `qwen2_5_omni_token2wav.py` — Token2Wav 模型（编码→梅尔→波形）

## 文件概述

本文件实现了 Qwen2.5-Omni 的 Token2Wav 模型，即三阶段流水线的最后阶段。它包含两个核心子模型：DiT（Diffusion Transformer）将 codec tokens 转换为梅尔频谱，BigVGAN 声码器将梅尔频谱转换为音频波形。文件还实现了 RK4 ODE 求解器用于扩散采样。

## 关键代码解析

### 1. DiT 子模型

```python
class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)
        self.text_embed = DiTCodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = DiTInputEmbedding(config)
        self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config.head_dim)
        self.transformer_blocks = nn.ModuleList([...])
```

DiT 模型使用时间步嵌入、codec 文本嵌入、旋转位置编码和自适应 LayerNorm。

### 2. ODE 采样

```python
class RungeKutta4ODESolver:
    def _rk4_step(self, function, time_start, time_step, time_end, value_start, ...):
        k1 = function_value_start or function(time_start, value_start)
        k2 = function(time_start + time_step / 3, value_start + time_step * k1 / 3)
        k3 = function(time_start + 2*time_step / 3, ...)
        k4 = function(time_end, ...)
        return (k1 + 3*(k2+k3) + k4) * time_step / 8
```

使用四阶 Runge-Kutta 方法求解 Flow Matching ODE，从噪声状态到梅尔频谱。

### 3. BigVGAN 声码器

```python
class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    def forward(self, mel_spectrogram):
        processed = self.process_mel_spectrogram(mel_spectrogram)
        hidden = self.conv_pre(processed)
        for layer_index in range(self.num_upsample_layers):
            hidden = self.ups[layer_index][0](hidden)
            residual = sum(self.resblocks[...](hidden)) / self.num_residual_blocks
            hidden = residual
        return torch.clamp(self.conv_post(self.activation_post(hidden)), -1.0, 1.0)
```

使用 SnakeBeta 激活函数和 AMPBlock 残差块的声码器。

### 4. 流式分块处理

```python
class Qwen2_5OmniToken2WavModel:
    def process_little_chunk(self, conditioning, reference_mel, codec_all, y_all, i, steps, ...):
        # 滑动窗口: past_cache_size + chunk_size + future_cache_size
        start_index = max(i * self.chunk_size - self.past_cache_size, 0)
        end_index = min((i+1) * self.chunk_size + self.future_cache_size, ...)
        # 使用 fast_block_sample 进行分块 ODE 采样
        generated = self.process_chunk_dit_batch(...)
        return self._process_chunk_for_50hz(...)
```

流式处理：使用过去和未来上下文窗口避免块间不连续。

### 5. vLLM 适配包装器

```python
class Qwen2_5OmniToken2WavForConditionalGenerationVLLM(nn.Module, SupportsPP):
    def load_weights(self, weights, spk_dict_path):
        # 1. 分离 buffers 和 parameters
        buffers = self.find_all_registers()
        weights_to_load = self.remove_buffers_from_weights(weights, buffers)
        # 2. 加载 parameters
        loaded = self.load_weights_without_buffers(weights_to_load)
        # 3. 重新加载 buffers
        loaded_buffers = self.reload_buffers_to_model(buffers)
        # 4. 加载说话人字典
        self.spk_dict = torch.load(spk_dict_path)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen2_5OmniToken2WavDiTModel` | 类 | DiT 扩散模型 |
| `Qwen2_5OmniToken2WavBigVGANModel` | 类 | BigVGAN 声码器 |
| `Qwen2_5OmniToken2WavModel` | 类 | DiT + BigVGAN 组合 |
| `Qwen2_5OmniToken2WavForConditionalGenerationVLLM` | 类 | vLLM 适配包装器 |
| `RungeKutta4ODESolver` | 类 | RK4 ODE 求解器 |
| `SnakeBeta` | 类 | Snake 激活函数 |
| `AMPBlock` | 类 | 自适应混合精度残差块 |
| `DiTDecoderLayer` | 类 | DiT Transformer 层 |

## 与其他模块的关系

- **被引用**: `qwen2_5_omni.py` 通过架构名 `"Qwen2_5OmniToken2WavModel"` 实例化
- **依赖**: `audio_length.py` 中的 `cap_and_align_mel_length` 和 `resolve_max_mel_frames`
- **依赖**: HuggingFace `Qwen2_5OmniPreTrainedModel` 基类
- **上游**: 接收 Talker 生成的 codec tokens

## 总结

Token2Wav 是 Qwen2.5-Omni 语音合成的最后一环。它采用 Flow Matching 扩散模型（DiT）生成梅尔频谱，再通过 BigVGAN 声码器合成波形。关键技术包括：RK4 ODE 求解、Classifier-Free Guidance、流式分块处理（支持 50Hz 采样率对齐）、以及 buffer 和 parameter 分离的权重加载策略。
