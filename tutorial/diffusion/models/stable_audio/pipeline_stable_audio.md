# `pipeline_stable_audio.py` -- Stable Audio 推理管线

## 文件概述

实现 `StableAudioPipeline`，文本到音频生成的完整管线。使用 T5 编码器进行文本编码，AutoencoderOobleck 进行音频编解码，CosineDPMSolverMultistepScheduler 进行采样。实现了 `SupportAudioOutput` 接口。

**文件路径**: `vllm_omni/diffusion/models/stable_audio/pipeline_stable_audio.py`

## 关键代码解析

### 管线组件

```python
class StableAudioPipeline(nn.Module, SupportAudioOutput):
    def __init__(self, ...):
        self.tokenizer = T5TokenizerFast.from_pretrained(...)
        self.text_encoder = T5EncoderModel.from_pretrained(...)
        self.vae = AutoencoderOobleck.from_pretrained(...)    # 音频 VAE
        self.projection_model = StableAudioProjectionModel.from_pretrained(...)  # 条件投影
        self.transformer = StableAudioDiTModel(...)
        self.scheduler = CosineDPMSolverMultistepScheduler.from_pretrained(...)
```

### 时长条件编码

```python
def encode_duration(self, audio_start_in_s, audio_end_in_s, ...):
    projection_output = self.projection_model(
        start_seconds=audio_start_in_s, end_seconds=audio_end_in_s)
    return seconds_start_hidden_states, seconds_end_hidden_states
```

音频时长条件通过 `StableAudioProjectionModel` 编码为嵌入向量，用于控制生成音频的长度。

### 去噪循环

```python
for t in timesteps:
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    noise_pred = self.transformer(
        latent_model_input, t.unsqueeze(0),
        encoder_hidden_states=text_audio_duration_embeds,
        global_hidden_states=audio_duration_embeds,
        rotary_embedding=rotary_embedding, ...)
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
```

### 音频后处理

```python
audio = self.vae.decode(latents_for_vae).sample
audio = audio[:, :, waveform_start:waveform_end]  # 裁剪到请求的时间段
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `StableAudioPipeline` | nn.Module | 文本到音频管线 |
| `encode_prompt()` | 方法 | T5 文本编码 + 投影 |
| `encode_duration()` | 方法 | 时长条件编码 |
| `get_stable_audio_post_process_func()` | 工厂函数 | 音频后处理 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `stable_audio_transformer.py` | DiT 模型 |
| 实现 | `interface.SupportAudioOutput` | 音频输出能力 |
| 依赖 | diffusers | AutoencoderOobleck、调度器、投影模型 |

## 总结

`StableAudioPipeline` 实现了完整的文本到音频流程：T5 文本编码 -> 时长条件编码 -> 条件拼接 -> DiT 去噪 -> Oobleck VAE 解码。支持通过 `audio_start_in_s` 和 `audio_end_in_s` 控制生成音频的时间段，以及 CFG 引导控制生成质量。
