# `pipeline_bagel.py` -- Bagel 生成管线

## 文件概述

本文件实现了 `BagelPipeline`，一个自包含的 Bagel 图像生成管线，集成了 LLM（Qwen2-MoT）、ViT（SigLIP）和 VAE。该管线支持文本到图像和图像到图像两种生成模式，并实现了双重 CFG（Classifier-Free Guidance）——同时对文本条件和图像条件进行无条件引导。

**文件路径**: `vllm_omni/diffusion/models/bagel/pipeline_bagel.py`

## 关键代码解析

### BagelGenParams 生成参数

```python
@dataclass
class BagelGenParams:
    num_timesteps: int = 50
    timestep_shift: float = 3.0
    cfg_text_scale: float = 4.0     # 文本条件 CFG 比例
    cfg_img_scale: float = 1.5      # 图像条件 CFG 比例
    cfg_interval: tuple = (0.4, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
```

双重 CFG 是 Bagel 的特色——分别对文本和图像条件进行无条件引导，需要维护三组 KV 缓存。

### BagelPipeline 初始化

```python
class BagelPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # 1. 加载 LLM 配置和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, ...)
        # 2. 加载 ViT（SigLIP）
        self.vit_model = SiglipVisionModel(vit_conf)
        self.vit_model = SiglipNaViTWrapper(self.vit_model)
        # 3. 创建 LLM
        self.language_model = Qwen2MoTForCausalLM(llm_config)
        # 4. 创建 VAE
        self.vae = AutoEncoder(ae_params)
        # 5. 创建 Bagel 核心模型
        self.bagel = Bagel(language_model=..., vit_model=..., config=...)
```

### SiglipNaViTWrapper ViT 包装器

```python
class SiglipNaViTWrapper(nn.Module):
    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        # 使用 NaViT 风格的 packed 输入处理变分辨率图像
        x = F.linear(packed_pixel_values, w, patch_embed.bias)
        pos = self.vision_model.embeddings.position_embedding(packed_flattened_position_ids)
        # 构建块对角注意力 mask
        mask = torch.full(..., torch.finfo(x.dtype).min, ...)
        for i in range(len(cu_seqlens_list) - 1):
            mask[..., start:end, start:end] = 0.0
```

支持 NaViT（可变分辨率）风格的 packed 图像输入。

### forward 主推理流程

```python
@torch.inference_mode()
def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
    # 1. 准备三组上下文（gen/cfg_text/cfg_img）
    gen_context = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(...)}
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)

    # 2. 如有图像输入，进行 VAE 和 ViT prefill
    if image_input:
        gen_context["past_key_values"] = self.bagel.forward_cache_update_vae(...)
        gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(...)

    # 3. 文本 prompt prefill
    gen_context["past_key_values"] = self.bagel.forward_cache_update_text(...)

    # 4. 准备三组 CFG 输入
    # gen: 完整条件; cfg_text: 无文本条件; cfg_img: 无图像条件

    # 5. 执行图像生成
    latents = self.bagel.generate_image(
        past_key_values=gen_context["past_key_values"],
        cfg_text_past_key_values=cfg_text_context["past_key_values"],
        cfg_img_past_key_values=cfg_img_context["past_key_values"],
        ...
    )

    # 6. VAE 解码
    img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)
```

### 权重加载

```python
def load_weights(self, weights):
    # 处理权重名称映射：
    # - "vae_model.*" -> "vae.*"
    # - "encoder.*"/"decoder.*" -> "vae.encoder.*"/"vae.decoder.*"
    # - 顶层 Bagel 核心层 -> "bagel.*"
    # 处理 QKV 融合投影映射
    # 处理 SigLIP 展平 patch embedding 的 reshape
```

权重加载器实现了复杂的名称映射和形状适配，处理不同格式检查点之间的兼容性。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `BagelPipeline` | nn.Module | 完整的 Bagel 生成管线 |
| `BagelGenParams` | dataclass | 生成参数（含双重 CFG） |
| `SiglipNaViTWrapper` | nn.Module | SigLIP ViT 的 NaViT 包装器 |
| `add_special_tokens()` | 函数 | 为 tokenizer 添加特殊 token |
| `get_bagel_post_process_func()` | 函数 | 获取后处理函数（Bagel 直接返回 PIL.Image） |
| `default_ae_params()` | 函数 | 默认 VAE 参数 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `bagel_transformer.py` | 使用 Bagel、NaiveCache、Qwen2MoTForCausalLM |
| 依赖 | `autoencoder.py` | 使用 AutoEncoder 进行图像编解码 |
| 依赖 | `DiffusersPipelineLoader` | 权重加载源 |
| 接口 | `OmniDiffusionRequest` | 接收标准化的扩散请求 |
| 输出 | `DiffusionOutput` | 返回标准化的生成结果 |

## 总结

`BagelPipeline` 是一个功能完整的多模态生成管线，其最大特色是双重 CFG 机制——同时维护三组 KV 缓存分别对应完全条件、无文本条件和无图像条件。管线支持注入预计算的 KV 缓存（用于与 vLLM AR 阶段配合的多阶段推理），也支持独立运行的图像和文本 prefill 流程。权重加载器处理了多种检查点格式的兼容性问题。
