# glm_image/ -- GLM-Image 模型目录索引

## 目录概述

GLM-Image 是一个两阶段文本到图像模型：AR 阶段（vLLM）生成 prior token，DiT 阶段使用 prior token 作为条件进行扩散去噪。支持图像编辑（通过 KV 缓存机制）。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化 |
| [`glm_image_transformer.py`](glm_image_transformer.md) | DiT 模型 + KV 缓存系统 |
| [`pipeline_glm_image.py`](pipeline_glm_image.md) | 两阶段推理管线 + CFG 并行 |

## 两阶段流程

```
阶段 1: AR (vLLM)
  文本 prompt -> vLLM GLM-Image AR -> prior_token_ids
                                       (+ prior_token_image_ids for i2i)

阶段 2: DiT (本目录)
  prior_token_ids + 字形嵌入 + 噪声潜变量
    -> GlmImageTransformer2DModel (28 层)
    -> AutoencoderKL 解码
    -> PIL Image
```

## 图像编辑 KV 缓存机制

```
1. WRITE 模式: 条件图像 -> VAE 编码 -> Transformer(t=0) -> 存储所有层 KV
2. READ 模式: 去噪循环中，每层注意力拼接缓存的条件 KV
3. CLEAR: 清除缓存
```

## 核心特色

1. **Prior Token 条件**: AR 模型生成的离散 token 嵌入为连续条件
2. **KV 缓存图像编辑**: WRITE/READ/SKIP 三模式
3. **ByT5 字形嵌入**: 字节级编码支持精确文本渲染
4. **CFG 并行**: rank 0 正向 + rank 1 负向（prior_token_drop）
5. **12 参数 AdaLN**: 同时调制图像流和文本流的 attention 和 FFN

## 总结

GLM-Image 的两阶段设计让 AR 模型负责高层语义理解（生成 prior token），DiT 模型负责低层视觉细节生成。KV 缓存机制实现了高效的图像编辑——条件图像只需编码一次，多步去噪中复用其 KV 状态。
