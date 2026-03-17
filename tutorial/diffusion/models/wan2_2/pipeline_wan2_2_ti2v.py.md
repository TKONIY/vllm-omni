# `pipeline_wan2_2_ti2v.py` — Wan2.2 文本+图像到视频管线

## 文件概述

本文件实现了 Wan2.2 的文本+图像到视频（TI2V）生成管线 `Wan22TI2VPipeline`。TI2V 结合了文本描述和参考图像的双重条件，生成内容与文本描述一致、风格与参考图像匹配的视频。

## 关键代码解析

### 1. 预处理

```python
def get_wan22_ti2v_pre_process_func(od_config):
    def pre_process_func(request):
        # 从请求中提取参考图像和文本描述
        # 编码参考图像为 CLIP 特征和 VAE 潜在表示
        # 自动调整视频帧数
        ...
    return pre_process_func
```

### 2. 管线结构

```python
class Wan22TI2VPipeline(nn.Module, SupportImageInput, CFGParallelMixin, ProgressBarMixin):
    def __init__(self, *, od_config, prefix=""):
        # 与 I2V 管线类似，包含 CLIP 图像编码器
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(...)

    def forward(self, req, ...):
        # 双重条件注入：
        # 1. 文本编码 -> 交叉注意力
        # 2. 图像编码 -> 图像交叉注意力
        # 3. 图像 VAE 潜在表示 -> 首帧条件
```

### 3. 首帧条件

TI2V 的一个关键特性是将参考图像的 VAE 潜在表示作为首帧的条件信息注入去噪过程，使生成视频的首帧与参考图像一致。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_wan22_ti2v_pre_process_func` | 函数 | TI2V 预处理工厂 |
| `get_wan22_ti2v_post_process_func` | 函数 | TI2V 后处理工厂 |
| `Wan22TI2VPipeline` | 类 | TI2V 管线 |

## 与其他模块的关系

- 与 `Wan22I2VPipeline` 共享 CLIP 图像编码器逻辑
- 使用同一个 `WanTransformer3DModel`
- 额外使用 VAE 编码参考图像作为首帧条件

## 总结

`pipeline_wan2_2_ti2v.py` 实现了双条件（文本+图像）的视频生成，通过 CLIP 特征和首帧 VAE 潜在表示的双路注入，使生成的视频同时满足文本语义和参考图像视觉一致性要求。
