# `pipeline_helios.py` — Helios 推理管线

## 文件概述

本文件实现了 Helios 的完整推理管线 `HeliosPipeline`，支持文本到视频（T2V）、图像到视频（I2V）和视频到视频（V2V）三种生成模式。管线的核心特色是分块视频生成：将长视频拆分为多个 chunk，每个 chunk 利用多期历史上下文进行去噪，从而生成任意长度的视频。

## 关键代码解析

### 1. 辅助函数

```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

根据序列长度动态计算调度器的 shift 参数，使不同分辨率的视频获得合适的噪声调度。

```python
def optimized_scale(positive_flat, negative_flat):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm
    return st_star
```

实现 CFG Zero Star 的最优缩放因子计算，通过投影最小化条件和无条件预测之间的差异。

### 2. 管线初始化

```python
class HeliosPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    def __init__(self, *, od_config, prefix=""):
        # 加载组件
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
        text_enc_cfg.tie_word_embeddings = True  # 修复 embed_tokens 全零问题
        self.text_encoder = UMT5EncoderModel.from_pretrained(...)
        self.vae = AutoencoderKLWan.from_pretrained(...)
        self.transformer = create_transformer_from_config(transformer_config)
        self.scheduler = HeliosScheduler(**scheduler_kwargs)
```

管线集成了 UMT5 文本编码器、Wan VAE、Helios Transformer 和统一调度器。注意 `tie_word_embeddings=True` 的修复：Helios 的检查点将词嵌入权重存储在 `shared.weight` 下，需要强制共享以避免全零嵌入。

### 3. 分块去噪主循环

```python
def forward(self, req, ...):
    for k in range(num_latent_chunk):
        # 1. 从历史缓冲区提取短/中/长期历史
        latents_history_long, latents_history_mid, latents_history_short = ...

        # 2. 准备当前块的噪声
        latents = self.prepare_latents(...)

        # 3. 选择单阶段或金字塔多阶段去噪
        if not is_enable_stage2:
            latents = self._stage1_sample(...)
        else:
            latents = self._stage2_sample(...)

        # 4. 更新历史缓冲区
        history_latents = torch.cat([history_latents, latents], dim=2)

        # 5. VAE 解码当前块
        current_video = self.vae.decode(current_latents)
        history_video = torch.cat([history_video, current_video], dim=2)
```

### 4. 金字塔多阶段去噪

```python
def _stage2_sample(self, latents, pyramid_num_stages, ...):
    # 先下采样到最小金字塔层级
    for _ in range(pyramid_num_stages - 1):
        latents_flat = F.interpolate(latents_flat, size=(height, width), mode="bilinear") * 2

    for i_s in range(pyramid_num_stages):
        # 每个阶段进行去噪
        for idx, t in enumerate(timesteps):
            noise_pred = self.transformer(...)
            latents = self.scheduler.step(...)

        if i_s > 0:
            # 上采样到下一层级并添加块噪声
            latents_flat = F.interpolate(latents_flat, size=(height, width), mode="nearest")
            noise = self.sample_block_noise(...)
            latents = alpha * latents + beta * noise
```

金字塔策略从低分辨率开始去噪，逐步上采样并在层级间注入块噪声以消除伪影。

### 5. I2V / V2V 支持

```python
def prepare_image_latents(self, image, ...):
    latents = self.vae.encode(image).latent_dist.sample(generator)
    latents = (latents - latents_mean) * latents_std
    # 创建 fake_image_latents 用于历史缓冲区初始化
    fake_video = image.repeat(1, 1, min_frames, 1, 1)
    fake_latents_full = self.vae.encode(fake_video).latent_dist.sample(generator)
    return latents, fake_latents
```

I2V 模式将输入图像编码为潜在空间并作为历史缓冲区的初始帧；V2V 模式将整段视频编码并注入噪声。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `calculate_shift` | 函数 | 动态 shift 参数计算 |
| `optimized_scale` | 函数 | CFG Zero Star 最优缩放 |
| `load_transformer_config` | 函数 | 加载 Transformer JSON 配置 |
| `create_transformer_from_config` | 函数 | 从配置字典创建 Transformer |
| `get_helios_post_process_func` | 函数 | 获取视频后处理函数 |
| `get_helios_pre_process_func` | 函数 | 获取请求预处理函数 |
| `HeliosPipeline` | 类 | 完整推理管线 |
| `HeliosPipeline.forward` | 方法 | 主入口，分块生成完整视频 |
| `HeliosPipeline._stage1_sample` | 方法 | 单阶段去噪循环 |
| `HeliosPipeline._stage2_sample` | 方法 | 金字塔多阶段去噪 |
| `HeliosPipeline.encode_prompt` | 方法 | 文本编码（正向+负向） |
| `HeliosPipeline.prepare_image_latents` | 方法 | I2V 图像编码 |
| `HeliosPipeline.prepare_video_latents` | 方法 | V2V 视频编码 |
| `HeliosPipeline.sample_block_noise` | 方法 | 金字塔层间块噪声生成 |

## 与其他模块的关系

- **`helios_transformer.py`**：调用 `HeliosTransformer3DModel` 进行噪声预测
- **`scheduling_helios.py`**：使用 `HeliosScheduler` 执行去噪步进
- **`CFGParallelMixin`**：继承自 `vllm_omni.diffusion.distributed.cfg_parallel`，支持 CFG 并行
- **`DiffusersPipelineLoader`**：通过 `weights_sources` 配置权重加载
- **`OmniDiffusionRequest`**：接收推理请求参数

## 总结

`pipeline_helios.py` 实现了 Helios 的核心推理逻辑，通过分块去噪 + 多期历史缓冲区实现任意长度视频的流式生成。支持 T2V/I2V/V2V 三种模式、单阶段与金字塔多阶段去噪策略、CFG Zero Star 以及 DMD 蒸馏模型加速。
