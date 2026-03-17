# `flux/__init__.py` -- FLUX.1 模型包初始化

## 文件概述

FLUX.1-dev 扩散模型子包的入口文件，导出核心的 Transformer 模型和 Pipeline 类。

**文件路径**: `vllm_omni/diffusion/models/flux/__init__.py`

## 导出内容

| 名称 | 类型 | 说明 |
|------|------|------|
| `FluxPipeline` | Pipeline 类 | FLUX.1 完整推理管线 |
| `FluxTransformer2DModel` | Transformer 类 | FLUX.1 双流/单流 Transformer |
| `get_flux_post_process_func` | 工厂函数 | 获取图像后处理函数 |

## 总结

FLUX.1 是 Black Forest Labs 推出的高质量文本到图像扩散模型，采用双流（Joint Attention）+ 单流 Transformer 架构。
