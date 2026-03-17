# `pipeline_longcat_image_edit.py` — LongCat 图像编辑管线

## 文件概述

该文件实现了 LongCat Image 的图像编辑管线，支持基于参考图像和文本指令的图像编辑功能。与文本到图像的管线不同，编辑管线额外处理输入图像的编码和条件注入，实现了 `SupportImageInput` 接口。

## 关键代码解析

### 预处理函数

```python
def get_longcat_image_edit_pre_process_func(od_config):
    def pre_process_func(request: OmniDiffusionRequest):
        # 自动计算输出尺寸（保持宽高比，目标面积 1024x1024）
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        # 预处理图像并存储到请求的 additional_information 中
        prompt["additional_information"]["preprocessed_image"] = image
        prompt["additional_information"]["prompt_image"] = prompt_image
```

预处理函数在请求进入管线前执行，自动处理图像尺寸计算和预处理。

### 视觉语言编码

```python
def _encode_prompt(self, prompt, image):
    # 使用 Qwen2.5-VL 的图像处理器
    raw_vl_input = self.image_processor_vl(images=image, return_tensors="pt")
    # 构建视觉-语言输入模板
    text = self.prompt_template_encode_prefix  # 包含 <|image_pad|> 占位符
    # 替换图像占位符为实际 token 数
    text = text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
    # 通过文本编码器获取联合表示
    text_output = self.text_encoder(input_ids=input_ids, pixel_values=pixel_values, ...)
```

编辑管线使用 Qwen2.5-VL 的视觉语言能力，将参考图像和编辑指令联合编码。

### 图像 latent 编码和拼接

```python
def prepare_latents(self, image, ...):
    image_latents = self._encode_vae_image(image=image, generator=generator)
    image_latents = self._pack_latents(image_latents, ...)
    # 图像 latent 使用 modality_id=2，与噪声 latent (modality_id=1) 区分
    image_latents_ids = prepare_pos_ids(modality_id=2, type="image", ...)
    return latents, image_latents, latents_ids, image_latents_ids
```

将参考图像编码为 latent 后与噪声 latent 拼接，使用不同的 modality_id 区分。

### 去噪循环中的图像条件

```python
for i, t in enumerate(timesteps):
    latent_model_input = latents
    if image_latents is not None:
        latent_model_input = torch.cat([latents, image_latents], dim=1)
    noise_pred = self.predict_noise_maybe_with_cfg(
        ..., output_slice=image_seq_len,  # 只取噪声 latent 对应的输出
    )
```

每步去噪时将图像 latent 拼接到输入中，但输出时只取噪声 latent 部分。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LongCatImageEditPipeline` | 类 | 图像编辑生成管线 |
| `get_longcat_image_edit_pre_process_func` | 函数 | 返回请求预处理函数 |
| `get_longcat_image_post_process_func` | 函数 | 返回后处理函数 |
| `calculate_dimensions` | 函数 | 根据面积和宽高比计算尺寸 |
| `retrieve_latents` | 函数 | 从 VAE 编码器输出提取 latent |

## 与其他模块的关系

- 复用 `LongCatImageTransformer2DModel` 作为核心模型
- 实现 `SupportImageInput` 接口，表明支持图像输入
- 继承 `CFGParallelMixin` 获得 CFG 并行能力
- 从 `pipeline_longcat_image` 导入 `calculate_shift` 函数
- 使用 Qwen2.5-VL 的视觉语言能力进行联合编码

## 总结

该管线是 LongCat Image 的图像编辑变体，核心区别在于：(1) 预处理阶段对输入图像进行 VAE 编码和尺寸计算；(2) 文本编码使用 VL 模型联合编码图像和文本；(3) 去噪过程中将参考图像 latent 与噪声 latent 拼接，使用不同的 modality_id 和位置编码。
