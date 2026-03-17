# `pipeline_wan2_2_i2v.py` — Wan2.2 图像到视频管线

## 文件概述

本文件实现了 Wan2.2 的图像到视频（I2V）生成管线 `Wan22I2VPipeline`。在基础 T2V 管线上增加了图像编码和注入逻辑，用户提供参考图像和文本描述即可生成对应的视频。

## 关键代码解析

### 1. 预处理

```python
def get_wan22_i2v_pre_process_func(od_config):
    def pre_process_func(request):
        # 从请求中提取参考图像
        # 根据 model_index.json 确定 CLIP 图像编码器
        # 调整图像尺寸并编码为 CLIP 特征
        ...
    return pre_process_func
```

预处理函数负责加载和编码参考图像为 CLIP 嵌入。

### 2. 管线初始化

```python
class Wan22I2VPipeline(nn.Module, SupportImageInput, CFGParallelMixin, ProgressBarMixin):
    def __init__(self, *, od_config, prefix=""):
        # 与基础管线相同的组件
        self.tokenizer = ...
        self.text_encoder = ...
        self.vae = ...
        self.transformer = ...

        # I2V 额外组件
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(...)
        self.image_processor = CLIPImageProcessor.from_pretrained(...)
```

增加了 CLIP 图像编码器用于提取参考图像特征。

### 3. 图像编码

```python
def encode_image(self, image):
    # 使用 CLIP 编码图像
    clip_output = self.image_encoder(image)
    image_embeds = clip_output.image_embeds
    # 或提取 last_hidden_state 作为图像条件
    return image_embeds
```

### 4. 去噪过程

```python
def forward(self, req, ...):
    # 1. 文本编码
    prompt_embeds = self.encode_prompt(...)

    # 2. 图像编码
    image_embeds = self.encode_image(image)

    # 3. 图像嵌入注入
    # 图像特征通过 WanImageEmbedding 注入到条件嵌入中

    # 4. 去噪循环（与 T2V 相同）
    latents = self.diffuse(prompt_embeds, ..., image_embeds=image_embeds, ...)

    # 5. VAE 解码
    video = self.vae.decode(latents)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_load_model_index` | 函数 | 加载 model_index.json |
| `get_wan22_i2v_pre_process_func` | 函数 | I2V 预处理工厂 |
| `get_wan22_i2v_post_process_func` | 函数 | I2V 后处理工厂 |
| `Wan22I2VPipeline` | 类 | I2V 管线 |

## 与其他模块的关系

- 继承 `SupportImageInput` 声明图像输入支持
- 使用与 T2V 管线相同的 `WanTransformer3DModel`（模型内置 I2V 交叉注意力）
- CLIP 图像编码器独立于 Transformer

## 总结

`pipeline_wan2_2_i2v.py` 在 T2V 基础上增加 CLIP 图像编码器，将参考图像特征注入 Transformer 的交叉注意力中实现图像到视频生成。
