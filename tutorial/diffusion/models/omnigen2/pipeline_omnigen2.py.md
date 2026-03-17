# `pipeline_omnigen2.py` — OmniGen2 图像生成管线

## 文件概述

该文件实现了 OmniGen2 的图像生成管线，支持文本到图像和图像编辑两种模式。管线整合了 Qwen2.5-VL 文本编码器、AutoencoderKL VAE、自定义的 Flow Matching 调度器和 OmniGen2 Transformer。支持图像输入条件和 Classifier-Free Guidance。

## 关键代码解析

### 自定义调度器

```python
class FlowMatchEulerDiscreteScheduler(ConfigMixin):
    def set_timesteps(self, num_inference_steps, device=None, shift=1.0):
        timesteps = torch.linspace(1, 1/num_inference_steps, num_inference_steps)
        self.sigmas = timesteps
    def step(self, model_output, timestep, sample):
        sigma = timestep
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output
```

### 视觉语言编码

```python
def _encode_prompt(self, prompt, images=None):
    # 使用 Qwen2.5-VL Processor 处理文本和图像
    inputs = self.processor(text=all_text, images=images, ...)
    text_output = self.text_encoder(
        input_ids=input_ids, pixel_values=pixel_values,
        image_grid_thw=image_grid_thw, output_hidden_states=True,
    )
    prompt_embeds = text_output.hidden_states[-1]
```

### 图像编辑支持

```python
def prepare_latents(self, image=None, ...):
    if image is not None:
        # 编码参考图像为 latent
        image_latents = self._encode_vae_image(image, generator)
        image_latents = self._pack_latents(image_latents, ...)
        # 创建条件掩码
        conditioning_mask = torch.ones(batch_size, 1, latent_seq_len)
```

### 去噪循环

```python
for i, t in enumerate(timesteps):
    # 拼接噪声 latent 和图像 latent
    if image_latents is not None:
        latent_model_input = torch.cat([latents, image_latents], dim=1)
    # Transformer 前向
    noise_pred = self.transformer(
        hidden_states=latent_model_input, timestep=t, ...)
    # CFG
    if do_cfg:
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
    # 调度器步进
    latents = self.scheduler.step(noise_pred, t, latents)[0]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniGen2Pipeline` | 类 | 图像生成/编辑管线 |
| `FlowMatchEulerDiscreteScheduler` | 类 | 自定义 Flow Matching 调度器 |
| `get_omnigen2_post_process_func` | 函数 | 获取后处理函数 |
| `get_omnigen2_pre_process_func` | 函数 | 获取预处理函数 |

## 与其他模块的关系

- 使用 `OmniGen2Transformer2DModel` 作为核心去噪模型
- 使用 Qwen2.5-VL 作为视觉语言文本编码器
- 实现 `SupportImageInput` 接口（支持图像输入）
- 使用 `DiffusersPipelineLoader` 加载权重

## 总结

OmniGen2Pipeline 支持文本到图像和图像编辑两种模式。通过 Qwen2.5-VL 的视觉语言能力实现多模态条件编码，图像编辑时将参考图像 latent 拼接到噪声 latent 中。管线使用自定义的 Flow Matching 调度器进行去噪。
