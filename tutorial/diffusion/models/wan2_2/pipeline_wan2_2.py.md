# `pipeline_wan2_2.py` — Wan2.2 文本到视频/图像管线

## 文件概述

本文件实现了 Wan2.2 的基础文本到视频/图像生成管线 `Wan22Pipeline`。使用 UMT5 文本编码器、Wan VAE 和 FlowMatch Euler 调度器，支持 CFG 并行和动态分辨率。

## 关键代码解析

### 1. 辅助函数

```python
def retrieve_latents(encoder_output, generator=None):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator=generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents

def load_transformer_config(model_path, subfolder="transformer", local_files_only=True):
    config_path = os.path.join(model_path, subfolder, "config.json")
    with open(config_path) as f:
        return json.load(f)

def create_transformer_from_config(config):
    # 将 JSON 配置映射到 WanTransformer3DModel 构造参数
    kwargs = {}
    for key in key_map:
        if key in config:
            val = config[key]
            if key in ("patch_size", "rope_dim") and isinstance(val, list):
                val = tuple(val)
            kwargs[key] = val
    return WanTransformer3DModel(**kwargs)
```

### 2. 管线初始化

```python
class Wan22Pipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    def __init__(self, *, od_config, prefix=""):
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(model, subfolder="text_encoder")
        self.vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae")
        self.transformer = create_transformer_from_config(transformer_config)
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=...)
```

### 3. 前向推理

```python
def forward(self, req, prompt=None, height=480, width=832, num_inference_steps=50, ...):
    # 1. 文本编码
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, ...)

    # 2. 准备潜在变量
    latents = self.prepare_latents(batch_size, num_channels_latents, ...)

    # 3. 设置时间步（动态 shift）
    sigmas = np.linspace(1.0, 0.0, num_steps + 1)
    mu = calculate_shift(image_seq_len, ...)
    timesteps, _ = retrieve_timesteps(self.scheduler, sigmas=sigmas, mu=mu)

    # 4. 去噪循环
    for i, t in enumerate(timesteps):
        noise_pred = self.predict_noise_maybe_with_cfg(do_true_cfg, ...)
        latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, ...)

    # 5. VAE 解码
    latents = latents / latents_std + latents_mean
    video = self.vae.decode(latents)
```

### 4. 预/后处理工厂

```python
def get_wan22_pre_process_func(od_config):
    def pre_process_func(request):
        # 自动计算帧数使其满足 VAE 对齐要求
        # 帧数公式: (num_latent_frames - 1) * temporal_scale + 1
        ...
    return pre_process_func

def get_wan22_post_process_func(od_config):
    video_processor = VideoProcessor(vae_scale_factor=8)
    def post_process_func(video, output_type="np"):
        return video_processor.postprocess_video(video, output_type=output_type)
    return post_process_func
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `retrieve_latents` | 函数 | VAE 潜在变量提取 |
| `load_transformer_config` | 函数 | 加载 Transformer 配置 |
| `create_transformer_from_config` | 函数 | 创建 Transformer 实例 |
| `get_wan22_post_process_func` | 函数 | 视频后处理工厂 |
| `get_wan22_pre_process_func` | 函数 | 请求预处理工厂 |
| `Wan22Pipeline` | 类 | T2V/T2I 管线 |
| `encode_prompt` | 方法 | UMT5 文本编码 |
| `prepare_latents` | 方法 | 潜在变量初始化 |

## 与其他模块的关系

- **`wan2_2_transformer.py`**：`WanTransformer3DModel` 核心去噪模型
- **`CFGParallelMixin`**：CFG 并行支持
- **`DiffusersPipelineLoader`**：权重加载
- **`pipeline_wan2_2_i2v.py`** / **`pipeline_wan2_2_ti2v.py`**：I2V/TI2V 变体

## 总结

`pipeline_wan2_2.py` 实现了 Wan2.2 的基础 T2V/T2I 推理管线，通过 UMT5 编码文本、FlowMatch Euler 动态调度和 CFG 并行完成高质量视频生成。是 I2V 和 TI2V 管线的基础参考。
