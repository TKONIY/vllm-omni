# `pipeline_flux.py` -- FLUX.1 推理管线

## 文件概述

本文件实现了 `FluxPipeline`，FLUX.1 模型的完整推理管线。管线包含双文本编码器（CLIP + T5）、VAE、调度器和 Transformer，支持 True CFG（Classifier-Free Guidance）和 CFG 并行。

**文件路径**: `vllm_omni/diffusion/models/flux/pipeline_flux.py`

## 关键代码解析

### 管线初始化

```python
class FluxPipeline(nn.Module, CFGParallelMixin):
    def __init__(self, *, od_config):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(...)
        self.text_encoder = CLIPTextModel.from_pretrained(...)     # CLIP 文本编码器
        self.text_encoder_2 = T5EncoderModel.from_pretrained(...)  # T5 文本编码器
        self.vae = AutoencoderKL.from_pretrained(...)
        self.transformer = FluxTransformer2DModel(od_config=od_config, ...)
        self.tokenizer = CLIPTokenizer.from_pretrained(...)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(...)
```

继承 `CFGParallelMixin` 支持 CFG 并行（多 GPU 分别计算正/负条件预测）。

### Latent 打包/解包

```python
@staticmethod
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    # FLUX 将潜变量打包为 2x2 patch: [B,C,H,W] -> [B,(H/2)*(W/2),C*4]
    latents = latents.view(batch_size, num_channels_latents, height//2, 2, width//2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height//2)*(width//2), num_channels_latents*4)
```

FLUX 将空间维度 2x2 的 patch 打包为序列 token。

### diffuse 去噪循环

```python
def diffuse(self, prompt_embeds, pooled_prompt_embeds, ..., do_true_cfg, ...):
    for i, t in enumerate(timesteps):
        positive_kwargs = {"hidden_states": latents, "timestep": timestep/1000, ...}
        if do_true_cfg:
            negative_kwargs = {... negative embeddings ...}
        # 使用 CFGParallelMixin 自动处理 CFG 并行
        noise_pred = self.predict_noise_maybe_with_cfg(
            do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, ...)
        latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
```

`predict_noise_maybe_with_cfg` 和 `scheduler_step_maybe_with_cfg` 来自 `CFGParallelMixin`，自动在单 GPU 和多 GPU CFG 并行之间切换。

### calculate_shift 动态时间步偏移

```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = image_seq_len * m + base_shift - m * base_seq_len
```

根据图像序列长度（与分辨率相关）动态计算时间步偏移。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `FluxPipeline` | nn.Module | 完整推理管线 |
| `diffuse()` | 方法 | 去噪循环 |
| `encode_prompt()` | 方法 | 双编码器文本编码 |
| `prepare_latents()` | 方法 | 准备初始噪声 |
| `get_flux_post_process_func()` | 工厂函数 | VAE 图像后处理 |
| `calculate_shift()` | 函数 | 动态时间步偏移计算 |
| `retrieve_timesteps()` | 函数 | 获取调度器时间步 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `flux_transformer.py` | Transformer 核心模型 |
| 混入 | `CFGParallelMixin` | CFG 并行支持 |
| 依赖 | diffusers | 调度器、VAE、图像处理器 |
| 依赖 | transformers | CLIP、T5 编码器 |

## 总结

`FluxPipeline` 是一个功能完整的文本到图像管线，其核心特色包括：(1) CLIP+T5 双编码器架构提供丰富的文本表示，(2) Latent 打包机制将 2x2 patch 转为序列 token，(3) True CFG 支持负向提示词引导，(4) 通过 `CFGParallelMixin` 实现多 GPU CFG 并行加速，(5) 动态时间步偏移根据分辨率自适应调整采样策略。
