# `pipeline_hunyuan_image_3.py` — HunyuanImage3 推理管线

## 文件概述

本文件实现了 HunyuanImage3 的完整推理管线 `HunyuanImage3Pipeline`，继承自 `HunyuanImage3PreTrainedModel` 和 `GenerationMixin`。管线整合了文本编码、图像编码、多模态序列构建、扩散去噪和 VAE 解码的完整流程，支持文本到图像生成和条件图像编辑。

## 关键代码解析

### 1. 管线初始化

```python
class HunyuanImage3Pipeline(HunyuanImage3PreTrainedModel, GenerationMixin):
    def __init__(self, od_config):
        self.model = HunyuanImage3Model(self.hf_config)
        self.vae = AutoencoderKLConv3D.from_config(self.hf_config.vae)
        self._tkwrapper = TokenizerWrapper(od_config.model)
        self.image_processor = HunyuanImage3ImageProcessor(self.hf_config)
        self.vision_model = Siglip2VisionModel(vision_config).vision_model
        self.vision_aligner = LightProjector(self.hf_config.vit_aligner)
        self.patch_embed = UNetDown(...)   # VAE -> Transformer
        self.final_layer = UNetUp(...)     # Transformer -> VAE
        self.timestep_emb = TimestepEmbedder(...)
        self.time_embed = TimestepEmbedder(...)
        self.time_embed_2 = TimestepEmbedder(...)
        self.lm_head = nn.Linear(...)
```

管线包含了完整的组件链：Tokenizer、Transformer、VAE、ViT 视觉编码器、投影器和多个时间步嵌入器。

### 2. 多模态 Token 实例化

```python
def instantiate_vae_image_tokens(self, x, images, ts, image_mask):
    # 将 VAE 编码的图像潜在表示嵌入到序列的 <img> 占位符位置
    t_emb = self.time_embed(t_i)
    image_i_seq, _, _ = self.patch_embed(image_i, t_i_emb)
    x[i].scatter_(dim=1, index=..., src=image_i_seq)

def instantiate_vit_image_tokens(self, x, cond_vit_images, cond_vit_image_mask, vit_kwargs):
    # 将 ViT 编码的条件图像特征嵌入到序列的条件图像占位符位置
    image_embed = self.vision_model(image, **cur_kwargs).last_hidden_state
    image_embed = self.vision_aligner(image_embed)
    x[i].scatter_(dim=1, index=..., src=image_embed)
```

序列中的各类占位符 token 通过 scatter 操作被替换为实际的嵌入向量。

### 3. 前向推理

```python
def forward_call(self, input_ids, ..., mode="gen_text", first_step=None, images=None, ...):
    inputs_embeds = self.model.embed_tokens(input_ids)

    if mode == "gen_image":
        if first_step:
            inputs_embeds, token_h, token_w = self.instantiate_vae_image_tokens(...)
            inputs_embeds = self.instantiate_timestep_tokens(...)
        else:
            # 后续步只需要 timestep + image tokens
            inputs_embeds = torch.cat([timestep_emb, image_emb], dim=1)

    # Transformer 前向
    with set_forward_context(None, self.vllm_config):
        outputs = self.model(inputs_embeds=inputs_embeds, custom_pos_emb=custom_pos_emb, ...)

    if mode == "gen_image":
        diffusion_prediction = self.ragged_final_layer(hidden_states, image_mask, timestep, ...)
```

首次去噪步需要完整序列（文本+图像），后续步利用 KV cache 只需传入图像和时间步 token。

### 4. 模型输入准备

```python
def prepare_model_inputs(self, prompt=None, mode="gen_image", ...):
    # 1. 文本 tokenization
    out = self._tkwrapper.apply_chat_template(...)
    # 2. 条件图像编码（VAE + ViT 双路径）
    cond_vae_images, cond_timestep, cond_vit_images = self._encode_cond_image(...)
    # 3. 2D RoPE 位置编码构建
    cos, sin = build_batch_2d_rope(image_infos=rope_image_info, ...)
    # 4. Attention mask 构建（文本因果 + 图像全连接）
```

### 5. 主入口 forward

```python
def forward(self, req, prompt="", image_size="auto", ...):
    model_inputs = self.prepare_model_inputs(prompt=prompt, mode="gen_image", ...)
    outputs = self._generate(**model_inputs)
    return DiffusionOutput(output=outputs[0])
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HunyuanImage3Pipeline` | 类 | 完整推理管线 |
| `instantiate_vae_image_tokens` | 方法 | 实例化 VAE 图像 token |
| `instantiate_vit_image_tokens` | 方法 | 实例化 ViT 条件图像 token |
| `instantiate_timestep_tokens` | 方法 | 实例化时间步 token |
| `prepare_model_inputs` | 方法 | 构建完整模型输入 |
| `forward_call` | 方法 | 单步前向推理 |
| `ragged_final_layer` | 方法 | 提取图像输出并反投影 |
| `vae_encode` | 方法 | VAE 编码 |
| `_generate` | 方法 | 调用扩散管线进行多步去噪 |

## 与其他模块的关系

- **`hunyuan_image_3_transformer.py`**：`HunyuanImage3Model` 和 `HunyuanImage3Text2ImagePipeline`
- **`hunyuan_image_3_tokenizer.py`**：`TokenizerWrapper` 构建多模态序列
- **`autoencoder.py`**：`AutoencoderKLConv3D` 编解码
- **Siglip2**：条件图像的视觉特征提取
- **`DiffusersPipelineLoader`**：权重加载

## 总结

`pipeline_hunyuan_image_3.py` 实现了 HunyuanImage3 的端到端推理流程，将自回归语言模型和扩散去噪有机结合。管线通过多模态 Tokenizer 将文本和图像统一编排为序列，利用 scatter 操作将 VAE 和 ViT 编码的图像特征注入占位符位置，最终通过 Transformer + UNet 反投影完成噪声预测。
