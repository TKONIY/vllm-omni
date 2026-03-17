# `bagel.py` — Bagel 多模态条件生成模型核心实现

## 文件概述

本文件实现了 Bagel 模型的 Omni 扩展版本，是整个 Bagel 模块的核心。主要包含：多模态处理器（图像预处理与 prompt 构建）、VAE 编码器（图像到潜变量）、以及基于 MoT（Mixture-of-Transformers）的前向传播路径。该模型支持图像理解（img2text）和图像生成（img2img）两种模式。

## 关键代码解析

### 1. 多模态处理器体系

```python
class OmniBagelProcessor(BagelProcessor):
    """扩展 HF 的 BagelProcessor，支持 img2img 模式下的图像预处理"""
    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)
        if is_img2img and images is not None:
            # img2img: 不做 resize，仅做 rescale
            image_kwargs["do_resize"] = False
            image_kwargs["do_rescale"] = True
```

img2img 模式下保留原始分辨率，因为 VAE 编码器需要原始尺寸信息来计算潜变量维度。

### 2. VAE 编码器

```python
class VAEEncoder(nn.Module):
    """轻量级 VAE 编码器，仅用于 AR 阶段的图像嵌入"""
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reg(self.encoder(x))           # 编码 + 对角高斯采样
        z = self.scale_factor * (z - self.shift_factor)  # 标准化
        return z
```

使用 Bagel 扩散管线中的 `Encoder` 和 `DiagonalGaussian`，将图像编码为标准化的潜变量表示。

### 3. img2img 嵌入处理

```python
def _process_img2img_input(self, multimodal_input):
    # 1. VAE 编码：图像 → 潜变量
    padded_latent = self.vae.encode(single_pv)
    # 2. Patch 化 + 位置编码 + 时间步编码
    vae_embeds = self.vae2llm(latent) + timestep_embeds + pos_embed
    # 3. ViT 编码：图像 → 视觉特征
    vit_emb = self._process_image_input({"pixel_values": vit_pixel_values})
    # 4. 组装：[start] + VAE嵌入 + [end] + [start] + ViT嵌入 + [end]
    combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
```

img2img 模式下，每张图像产生两组嵌入（VAE + ViT），用 vision_start/end 标记分隔。

### 4. MoT 前向传播

```python
def _mot_layer_forward(self, layer, positions, hidden_states, residual, vae_mask):
    """单层解码器的 MoT 路由前向传播"""
    # LayerNorm: VAE token 走 input_layernorm_moe_gen，其余走 input_layernorm
    normed[non_vae] = layer.input_layernorm(hidden_states[non_vae])
    normed[vae_mask] = layer.input_layernorm_moe_gen(hidden_states[vae_mask])
    # Attention: QKV/O 投影分别路由
    # MLP: VAE token 走 mlp_moe_gen，其余走 mlp
```

每一层 Transformer 解码器都根据 `vae_mask` 将 token 路由到不同的权重矩阵。

### 5. 位置 ID 调整

```python
def _adjust_positions_for_img2img(self, positions):
    # VAE tokens → position 0
    # separator → position 0
    # ViT tokens → position 1
    # text tokens → 2, 3, 4, ...
    new_positions[start : start + num_vae] = 0
    new_positions[vit_start : vit_start + num_vit] = 1
    new_positions[text_start:end] = torch.arange(2, 2 + num_text)
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `OmniBagelProcessor` | 类 | 扩展 BagelProcessor，支持 img2img |
| `OmniBagelProcessingInfo` | 类 | 处理信息，提供配置和数据解析 |
| `OmniBagelDummyInputsBuilder` | 类 | 构建性能分析用的虚拟输入 |
| `OmniBagelMultiModalProcessor` | 类 | 核心多模态处理器，处理 prompt 替换逻辑 |
| `OmniBagelDataParser` | 类 | 数据解析器，增加 img2img 模态支持 |
| `Img2ImgProcessorItems` | 类 | img2img 图像处理项 |
| `VAEEncoder` | 类 | 轻量级 VAE 编码器 |
| `OmniBagelForConditionalGeneration` | 类 | 核心模型类，集成 MoT 和 KV 传递 |

## 与其他模块的关系

- **继承自** `vllm.model_executor.models.bagel.BagelForConditionalGeneration`（vLLM 原生 Bagel 实现）
- **依赖** `vllm_omni.diffusion.models.bagel` 中的 `AutoEncoderParams`、`Encoder`、`DiagonalGaussian`、`PositionEmbedding`、`TimestepEmbedder`
- **使用** vLLM 的 `Qwen2DecoderLayer`、`Qwen2MLP` 等 Qwen2 模型组件
- **KV 元数据** 传递给 DiT 扩散阶段，用于图像去噪

## 总结

`bagel.py` 是一个复杂的多模态生成模型实现，核心创新在于 MoT 路由机制和位置编码对齐策略。通过在 AR 阶段同时编码 VAE 潜变量和 ViT 特征，并使用专用权重矩阵处理不同类型的 token，实现了 AR-DiT 两阶段图像生成管线中 KV cache 的无缝传递。
