# `pipeline_qwen_image_layered.py` — QwenImage 分层图像生成管线

## 文件概述

本文件实现了 QwenImage 的分层图像生成管线 `QwenImageLayeredPipeline`。分层生成允许用户通过多层图像输入控制生成结果，支持图层合成、遮罩和混合等高级编辑操作。

## 关键代码解析

### 1. 预处理

```python
def get_qwen_image_layered_pre_process_func(od_config):
    def pre_process_func(request):
        # 从请求中提取多层图像输入
        # 处理图层合成和遮罩
        # 计算目标分辨率
        ...
    return pre_process_func
```

### 2. 管线结构

```python
class QwenImageLayeredPipeline(nn.Module, SupportImageInput, QwenImageCFGParallelMixin):
    def forward(self, req, ...):
        # 1. 编码条件图像（多层）
        image_latents = self.encode_image(input_images)

        # 2. 文本编码
        prompt_embeds, ... = self.encode_prompt(...)

        # 3. 扩散循环
        latents = self.diffuse(..., image_latents=image_latents, ...)

        # 4. VAE 解码
        image = self.vae.decode(latents)
```

与编辑管线类似，但支持多层图像输入的处理和编码。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_qwen_image_layered_pre_process_func` | 函数 | 分层生成预处理工厂 |
| `calculate_dimensions` | 函数 | 尺寸计算 |
| `retrieve_timesteps` | 函数 | 时间步检索 |
| `retrieve_latents` | 函数 | VAE 潜在变量提取 |
| `QwenImageLayeredPipeline` | 类 | 分层图像生成管线 |

## 与其他模块的关系

- 继承 `SupportImageInput` 和 `QwenImageCFGParallelMixin`
- 使用与其他 QwenImage 管线相同的 Transformer 和 VAE 组件
- 预处理逻辑是区分于其他管线的核心差异

## 总结

`pipeline_qwen_image_layered.py` 实现了 QwenImage 的分层图像生成能力，通过多层图像输入和图层合成为用户提供了精细的图像控制能力，是编辑系列管线中最灵活的变体。
