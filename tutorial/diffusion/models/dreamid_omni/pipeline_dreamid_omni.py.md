# `pipeline_dreamid_omni.py` — DreamID-Omni 联合生成管线

## 文件概述

本文件实现了 DreamID-Omni 的完整推理管线 `DreamIDOmniPipeline`，支持视频和音频的联合生成。管线集成了 Wan VAE（视频）、MMAudio VAE（音频）、T5 文本编码器和 FusionModel，通过 CFG 并行实现高效推理。支持图像和音频两种条件输入。

## 关键代码解析

### 1. 管线初始化

```python
class DreamIDOmniPipeline(nn.Module, CFGParallelMixin, SupportImageInput, SupportAudioInput):
    def __init__(self, *, od_config, prefix=""):
        # VAE
        self.vae_model_video = init_wan_vae_2_2(model, rank=self.device)
        self.vae_model_audio = init_mmaudio_vae(model, rank=self.device)

        # 文本编码
        self.text_model = init_text_model(model, rank=self.device)

        # 融合模型
        Fusion_model = FusionModel(VIDEO_CONFIG, AUDIO_CONFIG)
        load_fusion_checkpoint(Fusion_model, checkpoint_path=...)
        self.model = Fusion_model
```

### 2. 视频/音频配置

```python
AUDIO_CONFIG = {
    "patch_size": [1], "model_type": "t2a", "dim": 3072, "num_heads": 24,
    "num_layers": 30, "in_dim": 20, "out_dim": 20, ...
}

VIDEO_CONFIG = {
    "patch_size": [1, 2, 2], "model_type": "ti2v", "dim": 3072, "num_heads": 24,
    "num_layers": 30, "in_dim": 48, "out_dim": 48, ...
}
```

视频和音频使用相同维度（3072）和层数（30）的 Transformer，但 patch_size 和通道数不同。

### 3. 条件加载

```python
def load_image_latent_ref_ip_video(self, images, audios, video_frame_height_width):
    # 加载参考图像的 IP 适配器特征
    # 编码参考音频为 MMAudio VAE 潜在表示
    # 编码参考图像为视频 VAE 潜在表示（首帧条件）
```

### 4. 前向推理

```python
def forward(self, req, prompt="", height=480, width=832, num_inference_steps=50, ...):
    # 1. 文本编码
    text_features = self.text_model.forward(prompt)

    # 2. 条件编码（图像 + 音频）
    image_latents, audio_latents, ip_features = self.load_image_latent_ref_ip_video(...)

    # 3. 准备噪声
    vid_latents = randn_tensor(video_shape, ...)
    audio_latents = randn_tensor(audio_shape, ...)

    # 4. 联合去噪循环
    for t in timesteps:
        vid_pred, audio_pred = self.model(
            vid=vid_latents, audio=audio_latents, t=t,
            vid_context=text_features, audio_context=text_features, ...)

        vid_latents = self.scheduler_video.step(vid_pred, t, vid_latents)
        audio_latents = self.scheduler_audio.step(audio_pred, t, audio_latents)

    # 5. VAE 解码
    video = self.vae_model_video.decode(vid_latents)
    audio = self.vae_model_audio.decode(audio_latents)
```

### 5. CFG 处理

```python
# 视频和音频各自使用不同的 CFG 缩放系数
self.video_cfg_scale = 3.0
self.video_ref_cfg_scale = 1.5
self.audio_cfg_scale = 4.0
self.audio_ref_cfg_scale = 2.0
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `AUDIO_CONFIG` | 字典 | 音频模型配置 |
| `VIDEO_CONFIG` | 字典 | 视频模型配置 |
| `DreamIDOmniPipeline` | 类 | 联合生成管线 |
| `load_image_latent_ref_ip_video` | 方法 | 条件编码（图像+音频） |
| `forward` | 方法 | 主推理入口 |

## 与其他模块的关系

- **`fusion.py`**：`FusionModel` 核心联合去噪模型
- **`wan2_2.py`**：`WanModel` 提供视频/音频基础 Transformer
- **`CFGParallelMixin`**：CFG 并行支持
- **`SupportImageInput` / `SupportAudioInput`**：声明多模态输入支持
- **`dreamid_omni` 外部包**：VAE 初始化、文本编码、检查点加载等工具

## 总结

`pipeline_dreamid_omni.py` 实现了 DreamID-Omni 的端到端视频+音频联合生成流程。通过共享时间步的同步去噪和 FusionModel 的双向交叉注意力，确保生成的视频和音频在内容和时序上保持一致。管线支持图像和音频条件输入，使用独立的 CFG 缩放系数和调度器分别控制两个模态。
