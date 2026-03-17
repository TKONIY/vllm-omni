# `pipeline_ltx2.py` — LTX2 文本到视频生成管线

## 文件概述

该文件实现了 LTX2 的文本到视频+音频联合生成管线，整合了 Gemma3 文本编码器、视频 VAE、音频 VAE、声码器（vocoder）和 LTX2 Transformer。管线支持 Classifier-Free Guidance、CFG 并行推理以及视频+音频的同步生成。

## 关键代码解析

### 组件初始化

```python
class LTX2Pipeline(nn.Module, CFGParallelMixin):
    def __init__(self, *, od_config, prefix=""):
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(...)
        self.vae = AutoencoderKLLTX2Video.from_pretrained(...)
        self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(...)
        self.vocoder = LTX2Vocoder.from_pretrained(...)
        self.connectors = LTX2TextConnectors.from_pretrained(...)
        self.transformer = LTX2VideoTransformer3DModel(od_config=od_config)
```

### 文本编码

```python
def encode_prompt(self, prompt, ...):
    text_inputs = self.tokenizer(prompt, padding="max_length", ...)
    prompt_embeds = self.text_encoder(input_ids=text_input_ids, ...)
    prompt_embeds = prompt_embeds.hidden_states[-1]
```

使用 Gemma3 模型进行文本编码。

### 视频/音频 latent 打包

```python
@staticmethod
def _pack_latents(latents, spatial_patch_size, temporal_patch_size):
    # 将 5D latent (B,C,T,H,W) 打包为 3D (B, num_patches, patch_dim)
```

### 双模态去噪

```python
noise_pred_video, noise_pred_audio = self.predict_noise_av_maybe_with_cfg(
    do_true_cfg=True, true_cfg_scale=guidance_scale,
    positive_kwargs=positive_kwargs, negative_kwargs=negative_kwargs,
)
```

使用 `predict_noise_av_maybe_with_cfg` 同时预测视频和音频的噪声。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `LTX2Pipeline` | 类 | 文本到视频+音频管线 |
| `get_ltx2_post_process_func` | 函数 | 获取后处理函数 |
| `load_transformer_config` | 函数 | 加载 Transformer 配置 |
| `create_transformer_from_config` | 函数 | 从配置创建 Transformer |
| `calculate_shift` | 函数 | 计算 timestep shift |

## 与其他模块的关系

- 使用 `LTX2VideoTransformer3DModel` 作为核心去噪模型
- 继承 `CFGParallelMixin` 获得 CFG 并行能力
- 使用 diffusers 的 LTX2 VAE 和声码器组件
- 使用 Gemma3 作为文本编码器

## 总结

LTX2Pipeline 实现了视频+音频的联合生成管线，是 vllm-omni 中功能最丰富的视频生成管线之一。其特色包括双模态（视频+音频）同步生成、CFG 并行推理、以及通过 LTX2TextConnectors 实现的文本条件处理。
