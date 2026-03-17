# `pipeline_ovis_image.py` — Ovis Image 生成管线

## 文件概述

该文件实现了 Ovis Image 的文本到图像生成管线。管线使用 Qwen3 (非 VL) 作为文本编码器、Qwen2 tokenizer、AutoencoderKL VAE 和 OvisImageTransformer2DModel。支持 Classifier-Free Guidance，并通过 `CFGParallelMixin` 实现 CFG 并行。

## 关键代码解析

### 文本编码 — Qwen3 模型

```python
class OvisImagePipeline(nn.Module, CFGParallelMixin):
    def __init__(self, ...):
        self.text_encoder = Qwen3Model.from_pretrained(model, subfolder="text_encoder")
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model, subfolder="tokenizer")
        self.system_prompt = "Describe the image by detailing the color, quantity, text, ..."
```

使用 Qwen3 (基座模型) 而非 VL 模型进行文本编码。

### 系统提示和 chat template

```python
def _get_messages(self, prompt):
    message = [{"role": "user", "content": self.system_prompt + each_prompt}]
    message = self.tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
```

使用 Qwen3 的 chat template 格式化提示词，禁用思考模式。

### 提示词嵌入提取

```python
def _get_ovis_prompt_embeds(self, prompt, ...):
    outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    prompt_embeds = outputs.last_hidden_state
    prompt_embeds = prompt_embeds * attention_mask[..., None]  # 掩码 padding
    prompt_embeds = prompt_embeds[:, self.user_prompt_begin_id:, :]  # 去除系统前缀
```

取最后一层隐藏状态，跳过前 28 个 token（系统前缀）。

### 去噪循环 — diffuse 方法

```python
def diffuse(self, latents, timesteps, prompt_embeds, negative_prompt_embeds, ...):
    self.scheduler.set_begin_index(0)
    for i, t in enumerate(timesteps):
        noise_pred = self.predict_noise_maybe_with_cfg(
            do_true_cfg, guidance_scale, positive_kwargs, negative_kwargs, cfg_normalize,
        )
        latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
    return latents
```

将去噪循环抽取为独立方法，使用 `CFGParallelMixin` 的辅助方法。

### 主前向传播

```python
def forward(self, req, ...):
    # 1. 编码正向/负向提示词
    prompt_embeds, text_ids = self.encode_prompt(prompt=prompt, ...)
    negative_prompt_embeds, negative_text_ids = self.encode_prompt(prompt=negative_prompt, ...)
    # 2. 准备 latent
    latents, latent_image_ids = self.prepare_latents(...)
    # 3. 准备时间步
    timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, image_seq_len)
    # 4. 去噪
    latents = self.diffuse(latents, timesteps, ...)
    # 5. VAE 解码
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    image = self.vae.decode(latents)[0]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OvisImagePipeline` | 类 | 图像生成管线 |
| `get_ovis_image_post_process_func` | 函数 | 获取后处理函数 |
| `calculate_shift` | 函数 | 计算 timestep shift |
| `retrieve_timesteps` | 函数 | 获取调度器时间步 |

## 与其他模块的关系

- 使用 `OvisImageTransformer2DModel` 作为核心去噪模型
- 继承 `CFGParallelMixin` 获得 CFG 并行能力
- 使用 Qwen3 作为文本编码器
- 使用 `FlowMatchEulerDiscreteScheduler` 调度器

## 总结

Ovis Image 管线是一个结构清晰的文本到图像管线，特色在于使用 Qwen3 基座模型（非 VL）作为文本编码器，通过系统提示引导模型生成图像描述性嵌入。管线将去噪循环抽取为 `diffuse` 方法，代码组织优于部分其他管线。
