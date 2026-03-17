# `flux2/__init__.py` -- Flux 2 模型包初始化

## 文件概述

Flux 2 扩散模型子包的入口文件，导出 Transformer 模型、Pipeline 和后处理函数。

**文件路径**: `vllm_omni/diffusion/models/flux2/__init__.py`

## 导出内容

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2Pipeline` | Pipeline 类 | Flux 2 推理管线（含 Mistral3 文本编码器） |
| `Flux2Transformer2DModel` | Transformer 类 | Flux 2 双流/单流 Transformer |
| `get_flux2_post_process_func` | 工厂函数 | 图像后处理函数 |

## 总结

Flux 2 是 FLUX.1 的升级版本，采用 SwiGLU 激活、全局 Modulation 参数共享和更大的模型维度。支持图像到图像（超分辨率）等高级功能。
