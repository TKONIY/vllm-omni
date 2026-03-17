# `glm_image/__init__.py` -- GLM-Image 模型包初始化

## 文件概述

GLM-Image 扩散模型子包入口，导出 Transformer、KV 缓存、Pipeline 及处理函数。

**文件路径**: `vllm_omni/diffusion/models/glm_image/__init__.py`

## 导出内容

| 名称 | 类型 | 说明 |
|------|------|------|
| `GlmImageTransformer2DModel` | Transformer 类 | GLM-Image DiT 模型 |
| `GlmImageKVCache` | 缓存类 | 图像编辑用 KV 缓存 |
| `GlmImagePipeline` | Pipeline 类 | 完整推理管线 |
| `get_glm_image_post_process_func` | 工厂函数 | 图像后处理 |
| `get_glm_image_pre_process_func` | 工厂函数 | 图像预处理 |

## 总结

GLM-Image 是一个两阶段模型：AR 阶段（vLLM）生成 prior token，DiT 阶段进行扩散去噪。支持文本到图像和图像编辑两种模式。
