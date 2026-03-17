# `pipeline_qwen_image_edit.py` — QwenImage 图像编辑管线

## 文件概述

本文件实现了 QwenImage 的图像编辑管线 `QwenImageEditPipeline`，在文本到图像管线的基础上增加了条件图像输入。用户提供原始图像和编辑指令，管线将原始图像的 VAE 潜在表示与噪声潜在表示拼接后送入 Transformer 进行去噪。

## 关键代码解析

### 1. 预处理

```python
def get_qwen_image_edit_pre_process_func(od_config):
    def pre_process_func(request):
        # 从请求中提取图像输入
        # 根据目标面积和宽高比计算输出尺寸
        # 调整输入图像大小
        ...
    return pre_process_func
```

预处理函数负责从请求中提取条件图像，并根据目标面积自动计算输出分辨率。

### 2. 管线结构

```python
class QwenImageEditPipeline(nn.Module, SupportImageInput, QwenImageCFGParallelMixin):
    def forward(self, req, ...):
        # 1. 编码条件图像
        image_latents = self.encode_image(input_image)

        # 2. 文本编码
        prompt_embeds, ... = self.encode_prompt(...)

        # 3. 扩散循环（传入 image_latents）
        latents = self.diffuse(..., image_latents=image_latents, ...)

        # 4. VAE 解码
        image = self.vae.decode(latents)
```

关键区别在于 `image_latents` 参数：编辑管线将条件图像的潜在表示拼接到噪声潜在表示上，使 Transformer 能同时看到条件信息和待去噪的数据。

### 3. 图像编码

```python
def encode_image(self, image):
    image_latents = retrieve_latents(self.vae.encode(image))
    image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
    return image_latents
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_qwen_image_edit_pre_process_func` | 函数 | 图像编辑预处理工厂 |
| `get_qwen_image_edit_post_process_func` | 函数 | 图像编辑后处理工厂 |
| `calculate_dimensions` | 函数 | 根据目标面积计算尺寸 |
| `QwenImageEditPipeline` | 类 | 图像编辑管线 |

## 与其他模块的关系

- 继承 `SupportImageInput` 接口声明图像输入支持
- 继承 `QwenImageCFGParallelMixin` 复用 `diffuse` 方法
- 使用与基础管线相同的 Transformer 和 VAE

## 总结

`pipeline_qwen_image_edit.py` 在基础文本到图像管线上增加了条件图像编码和拼接逻辑，实现了指令驱动的图像编辑功能。
