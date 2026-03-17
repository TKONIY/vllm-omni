# flux2/ -- Flux 2 模型目录索引

## 目录概述

Flux 2 是 FLUX.1 的升级版，采用 SwiGLU 激活、全局 Modulation 参数共享和 Mistral3 多模态编码器。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化 |
| [`flux2_transformer.py`](flux2_transformer.md) | SwiGLU + 全局 Modulation Transformer |
| [`pipeline_flux2.py`](pipeline_flux2.md) | Mistral3 编码器 + 图像超分 Pipeline |

## 相较 FLUX.1 的改进

| 方面 | FLUX.1 | Flux 2 |
|------|--------|--------|
| 激活函数 | GELU | SwiGLU |
| 调制方式 | 逐块 AdaLNZero | 全局 Modulation 共享 |
| RoPE | 3轴 (16+56+56) | 4轴 (32x4) |
| 文本编码器 | CLIP + T5 | Mistral3 (多模态) |
| 单流 MLP | 独立 proj_mlp | 融合 QKV+MLP 投影 |
| bias | 有 | 无 |

## 总结

Flux 2 通过全局 Modulation 减少参数量、SwiGLU 提升表达能力、融合 QKV+MLP 投影减少访存，在效率和质量上均有提升。
