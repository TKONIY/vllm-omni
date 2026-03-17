# `pipeline_glm_image.py` -- GLM-Image 推理管线

## 文件概述

实现 `GlmImagePipeline`，GLM-Image 的完整两阶段推理管线。AR 阶段（vLLM）生成 prior token IDs，DiT 阶段使用这些 token 作为条件进行扩散去噪。支持文本到图像和图像编辑（通过 KV 缓存机制）两种模式。

**文件路径**: `vllm_omni/diffusion/models/glm_image/pipeline_glm_image.py`

## 关键代码解析

### 预处理函数

```python
def get_glm_image_pre_process_func(od_config):
    def pre_process_func(request: OmniDiffusionRequest):
        # 对输入图像进行尺寸对齐
        multiple_of = vae_scale_factor * patch_size
        img_h = (img_h // multiple_of) * multiple_of
        img_w = (img_w // multiple_of) * multiple_of
        processed = image_processor.preprocess(img, height=img_h, width=img_w)
        # 存储到 request 的 additional_information 中
        prompt["additional_information"]["preprocessed_image"] = processed
```

### 字形文本提取

```python
def get_glyph_texts(self, prompt):
    # 提取引号内的文本用于字形渲染
    ocr_texts = (
        re.findall(r"'([^']*)'", prompt)
        + re.findall(r'\u201c([^\u201c\u201d]*)\u201d', prompt)
        + re.findall(r'"([^"]*)"', prompt)
        + re.findall(r'\u300c([^\u300c\u300d]*)\u300d', prompt)
    )
```

从 prompt 中提取引号内的文本，通过 ByT5 编码为字形嵌入，支持图像中的文本渲染。

### 图像编辑 KV 缓存准备

```python
def _prepare_condition_image_kv_cache(self, condition_images, prior_token_image_ids, ...):
    kv_caches = self.transformer.create_kv_cache()
    kv_caches.set_mode("write")
    for condition_image, condition_prior_token_id in zip(condition_images, prior_token_image_ids):
        condition_latent = retrieve_latents(self.vae.encode(condition_image.unsqueeze(0)), ...)
        _ = self.transformer(
            hidden_states=condition_latent,
            encoder_hidden_states=torch.zeros_like(prompt_embeds)[:1, :0, ...],
            prior_token_id=condition_prior_token_id,
            timestep=torch.zeros((1,), ...),
            kv_cache=kv_caches,
        )
    return kv_caches
```

图像编辑流程：
1. 对条件图像进行 VAE 编码
2. 以 timestep=0 运行 Transformer 并写入 KV 缓存
3. 切换为 READ 模式，在去噪循环中使用缓存

### diffuse 去噪循环（CFG 并行感知）

```python
def diffuse(self, latents, prior_token_id, prompt_embeds, negative_prompt_embeds, ...):
    if cfg_parallel_ready:
        cfg_rank = get_classifier_free_guidance_rank()
        if cfg_rank == 0:
            local_pred = self.transformer(encoder_hidden_states=prompt_embeds, prior_token_drop=False, ...)
        else:
            local_pred = self.transformer(encoder_hidden_states=negative_prompt_embeds, prior_token_drop=True, ...)
        gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
        if cfg_rank == 0:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, ...)
        cfg_group.broadcast(latents, src=0)
```

CFG 并行实现：rank 0 计算正向预测，rank 1 计算负向预测（prior_token_drop=True），然后 all_gather 聚合并在 rank 0 上应用 CFG 公式。

### forward 主流程

```python
def forward(self, req: OmniDiffusionRequest):
    # 1. 获取 prior tokens（从 AR 阶段或外部注入）
    prior_token_id = req.sampling_params.extra_args.get("prior_token_ids")
    # 2. 编码字形嵌入
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, ...)
    # 3. 准备 KV 缓存（图像编辑模式）
    if is_image_edit:
        kv_caches = self._prepare_condition_image_kv_cache(...)
        kv_caches.set_mode("read")
    # 4. 准备潜变量和时间步
    latents = self.prepare_latents(...)
    timesteps, _ = retrieve_timesteps(self.scheduler, ..., sigmas=sigmas, mu=mu)
    # 5. 去噪循环
    latents = self.diffuse(latents, prior_token_id, ..., kv_caches=kv_caches)
    # 6. VAE 解码（带 latents_mean/latents_std 归一化）
    latents = latents * latents_std + latents_mean
    image = self.vae.decode(latents, ...)[0]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `GlmImagePipeline` | nn.Module | 完整推理管线 |
| `diffuse()` | 方法 | CFG 并行去噪循环 |
| `encode_prompt()` | 方法 | ByT5 字形编码 |
| `get_glyph_texts()` | 方法 | 提取引号内文本 |
| `_prepare_condition_image_kv_cache()` | 方法 | 图像编辑 KV 缓存准备 |
| `get_glm_image_pre_process_func()` | 工厂函数 | 图像预处理 |
| `get_glm_image_post_process_func()` | 工厂函数 | 图像后处理 |
| `calculate_shift()` | 函数 | 动态时间步偏移 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `glm_image_transformer.py` | DiT 模型和 KV 缓存 |
| 依赖 | CFG 并行框架 | `get_cfg_group()` 等 |
| 输入 | vLLM AR 阶段 | 接收 `prior_token_ids` |
| 依赖 | transformers | ByT5Tokenizer、T5EncoderModel |

## 总结

`GlmImagePipeline` 实现了 GLM-Image 的两阶段生成流程。核心特色包括：(1) Prior token 条件——AR 模型生成的离散 token 作为图像条件注入 DiT，(2) 图像编辑通过 KV 缓存实现——先 WRITE 条件图像特征再 READ 用于去噪，(3) CFG 并行支持——rank 0/1 分别计算正/负预测并 all_gather 聚合，(4) ByT5 字形嵌入支持高质量文本渲染，(5) 带 latents_mean/std 归一化的 VAE 解码。
