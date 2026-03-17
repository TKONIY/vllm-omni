# `pipeline_sd3.py` -- Stable Diffusion 3 推理管线

## 文件概述

实现 `StableDiffusion3Pipeline`，SD3 的完整推理管线。使用三个文本编码器（CLIP x2 + T5）进行文本编码，支持 CFG 并行。

**文件路径**: `vllm_omni/diffusion/models/sd3/pipeline_sd3.py`

## 关键代码解析

### 三编码器文本编码

```python
class StableDiffusion3Pipeline(nn.Module, CFGParallelMixin):
    def __init__(self, ...):
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(...)    # CLIP 1
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(...)  # CLIP 2
        self.text_encoder_3 = T5EncoderModel.from_pretrained(...)              # T5
        self.tokenizer = CLIPTokenizer.from_pretrained(...)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(...)
        self.tokenizer_3 = T5Tokenizer.from_pretrained(...)
```

### encode_prompt 三路编码

```python
def encode_prompt(self, prompt, prompt_2, prompt_3, ...):
    # CLIP 1: hidden_states[-2] + pooled_output
    prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(prompt, clip_model_index=0)
    # CLIP 2: hidden_states[-2] + pooled_output
    prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(prompt_2, clip_model_index=1)
    # 拼接两个 CLIP 嵌入
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
    # T5: last_hidden_state
    t5_prompt_embed = self._get_t5_prompt_embeds(prompt_3, ...)
    # 对齐维度并拼接
    clip_prompt_embeds = F.pad(clip_prompt_embeds, (0, t5_dim - clip_dim))
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
```

### diffuse 去噪循环

```python
def diffuse(self, latents, timesteps, prompt_embeds, pooled_prompt_embeds, ...):
    for _, t in enumerate(timesteps):
        noise_pred = self.predict_noise_maybe_with_cfg(
            do_true_cfg, guidance_scale, positive_kwargs, negative_kwargs, ...)
        latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
```

### DistributedAutoencoderKL

```python
self.vae = DistributedAutoencoderKL.from_pretrained(model, subfolder="vae", ...)
```

使用分布式 VAE 支持大分辨率解码。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `StableDiffusion3Pipeline` | nn.Module | SD3 完整推理管线 |
| `encode_prompt()` | 方法 | 三编码器文本编码 |
| `diffuse()` | 方法 | 去噪循环 |
| `get_sd3_image_post_process_func()` | 工厂函数 | 图像后处理 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `sd3_transformer.py` | SD3 Transformer 模型 |
| 混入 | `CFGParallelMixin` | CFG 并行 |
| 依赖 | `DistributedAutoencoderKL` | 分布式 VAE |

## 总结

SD3 Pipeline 的核心特色是三编码器架构（2x CLIP + T5），将不同编码器的输出在序列和特征维度上拼接，提供丰富的文本表示。两个 CLIP 编码器提供 pooled 嵌入用于全局条件，T5 提供长文本的细粒度条件。通过 `CFGParallelMixin` 支持 CFG 并行加速。
