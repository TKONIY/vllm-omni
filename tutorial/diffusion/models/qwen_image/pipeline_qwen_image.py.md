# `pipeline_qwen_image.py` — QwenImage 文本到图像管线

## 文件概述

本文件实现了 QwenImage 的文本到图像生成管线 `QwenImagePipeline`，使用 Qwen2.5-VL 作为文本编码器，FlowMatch Euler 调度器进行去噪。管线支持动态分辨率、CFG 并行和自定义噪声调度。

## 关键代码解析

### 1. 辅助函数

```python
def apply_rotary_emb_qwen(hidden_states, freqs_cis):
    # QwenImage 风格的旋转位置编码应用
    ...

def get_timestep_embedding(timesteps, embedding_dim, ...):
    # 正弦/余弦时间步嵌入
    ...

def calculate_shift(image_seq_len, ...):
    # 根据序列长度动态计算 shift 参数
    ...
```

### 2. 管线初始化

```python
class QwenImagePipeline(nn.Module, QwenImageCFGParallelMixin):
    def __init__(self, *, od_config, prefix=""):
        # 文本编码器
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(...)

        # VAE（分布式版本）
        self.vae = DistributedAutoencoderKLQwenImage(...)

        # Transformer
        self.transformer = QwenImageTransformer2DModel(...)

        # 调度器
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=...)
```

### 3. 文本编码

```python
def encode_prompt(self, prompt, negative_prompt=None, ...):
    # 使用 Qwen2.5-VL 编码文本
    prompt_embeds = self.text_encoder.model(input_ids=text_input_ids, ...).last_hidden_state
    # 处理 attention mask
    prompt_embeds_mask = text_inputs.attention_mask
```

### 4. 主入口

```python
def forward(self, req, prompt=None, height=1024, width=1024, num_inference_steps=20, ...):
    # 1. 文本编码
    prompt_embeds, negative_prompt_embeds, ... = self.encode_prompt(...)

    # 2. 准备潜在变量
    latents = randn_tensor(latent_shape, ...)

    # 3. 设置时间步（动态 shift）
    mu = calculate_shift(latent_image_seq_len)
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, ..., mu=mu)

    # 4. 扩散循环（继承自 QwenImageCFGParallelMixin）
    latents = self.diffuse(prompt_embeds, ..., timesteps, ...)

    # 5. VAE 解码
    image = self.vae.decode(latents / self.vae.config.scaling_factor)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_qwen_image_post_process_func` | 函数 | 图像后处理函数工厂 |
| `calculate_shift` | 函数 | 动态 shift 计算 |
| `apply_rotary_emb_qwen` | 函数 | QwenImage RoPE |
| `QwenImagePipeline` | 类 | 文本到图像管线 |
| `encode_prompt` | 方法 | 文本编码 |
| `predict_noise` | 方法 | 单步噪声预测 |

## 与其他模块的关系

- **`qwen_image_transformer.py`**：`QwenImageTransformer2DModel` 核心去噪模型
- **`cfg_parallel.py`**：继承 `QwenImageCFGParallelMixin` 获得扩散循环
- **`autoencoder_kl_qwenimage.py`**：通过 `DistributedAutoencoderKLQwenImage` 使用
- **Qwen2.5-VL**：文本编码器

## 总结

`pipeline_qwen_image.py` 实现了 QwenImage 的文本到图像基础管线，集成 Qwen2.5-VL 文本编码器、动态分辨率调度和 CFG 并行支持，是其他编辑管线的基础参考实现。
