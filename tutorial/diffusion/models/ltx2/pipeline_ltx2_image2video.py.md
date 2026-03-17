# `pipeline_ltx2_image2video.py` — LTX2 图像到视频生成管线

## 文件概述

该文件实现了 LTX2 的图像到视频（Image-to-Video, I2V）生成管线，继承自 `LTX2Pipeline` 并扩展了图像条件处理。管线接收一张参考图像和文本提示，生成以该图像为首帧的视频及配套音频。

## 关键代码解析

### 图像条件 latent 准备

```python
def prepare_latents(self, image, batch_size, ...):
    # 编码图像为 latent
    init_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0).unsqueeze(2)), ...)]
    init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
    # 将首帧 latent 重复到所有帧
    init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)
    # 构建条件掩码：首帧=1（条件），其余帧=0（需生成）
    conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
    conditioning_mask[:, :, 0] = 1.0
    # 混合：条件帧保持 + 其余帧用噪声
    latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)
```

关键设计：通过 conditioning_mask 区分条件帧（首帧）和待生成帧。

### I2V 特定的时间步处理

```python
video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)
```

条件帧的时间步为 0（表示已确定），其余帧使用正常时间步。

### I2V 调度器步进

```python
def _step_video_latents_i2v(self, noise_pred_video, latents, t, ...):
    # 只对非首帧进行调度器步进
    noise_pred_video = noise_pred_video[:, :, 1:]
    noise_latents = latents_unpacked[:, :, 1:]
    pred_latents = self.scheduler.step(noise_pred_video, t, noise_latents, ...)[0]
    # 重新拼接首帧
    latents_unpacked = torch.cat([latents_unpacked[:, :, :1], pred_latents], dim=2)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LTX2ImageToVideoPipeline` | 类 | 图像到视频管线 |
| `get_ltx2_post_process_func` | 函数 | 复用父类的后处理函数 |

## 与其他模块的关系

- 继承 `LTX2Pipeline`，复用文本编码、音频处理等功能
- 通过 `support_image_input = True` 标识支持图像输入
- 从请求的 `multi_modal_data` 或 `additional_information` 中提取图像

## 总结

该管线是 LTX2Pipeline 的 I2V 扩展，核心差异在于：(1) 使用 conditioning_mask 机制保护首帧 latent 不被去噪修改；(2) 时间步乘以 `(1 - mask)` 使条件帧的时间步为零；(3) 调度器步进时跳过首帧。这种设计确保生成的视频与参考图像在首帧保持一致。
