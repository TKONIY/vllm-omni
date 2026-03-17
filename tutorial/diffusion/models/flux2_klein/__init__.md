# `flux2_klein/__init__.py` -- Flux 2 Klein 模型包初始化

## 文件概述

Flux 2 Klein 扩散模型子包的入口文件。Klein 是 Flux 2 的变体，增加了序列并行（Sequence Parallel）支持。

**文件路径**: `vllm_omni/diffusion/models/flux2_klein/__init__.py`

## 导出内容

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2KleinPipeline` | Pipeline 类 | Flux 2 Klein 推理管线 |
| `Flux2Transformer2DModel` | Transformer 类 | 支持序列并行的 Flux 2 Transformer |
| `get_flux2_klein_post_process_func` | 工厂函数 | 图像后处理函数 |

## 总结

Flux 2 Klein 在 Flux 2 基础上增加了 Ulysses/Ring 序列并行支持，适用于高分辨率图像生成的多 GPU 场景。
