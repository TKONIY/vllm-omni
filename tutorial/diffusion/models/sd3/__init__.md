# `sd3/__init__.py` -- Stable Diffusion 3 模型包初始化

## 文件概述

SD3 扩散模型子包入口，导出 Pipeline、Transformer 和后处理函数。

**文件路径**: `vllm_omni/diffusion/models/sd3/__init__.py`

## 导出内容

| 名称 | 类型 | 说明 |
|------|------|------|
| `StableDiffusion3Pipeline` | Pipeline 类 | SD3 推理管线 |
| `SD3Transformer2DModel` | Transformer 类 | MMDiT 架构 Transformer |
| `get_sd3_image_post_process_func` | 工厂函数 | 图像后处理 |

## 总结

Stable Diffusion 3 采用 MMDiT（Multimodal Diffusion Transformer）架构，使用三个文本编码器（2x CLIP + T5）。
