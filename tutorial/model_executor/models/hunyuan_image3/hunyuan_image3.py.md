# `hunyuan_image3.py` — HunyuanImage3 顶层模型

## 文件概述

HunyuanImage3 的完整 AR 模型实现，约 1600 行代码。基于 HunYuan MoE LLM，集成 Siglip2 视觉编码器和 3D VAE。支持文本到图像（t2i）和图像到图像（i2i）生成。

## 关键代码解析

### 模型架构

该模型继承自 vLLM 的 HunYuanModel，扩展了以下多模态组件：

- **Siglip2VisionTransformer**：视觉编码器，处理 i2i 源图像
- **LightProjector**：将视觉特征投影到 LLM 隐空间
- **AutoencoderKLConv3D**：3D VAE 用于潜空间编解码
- **MoE 层**：SharedFusedMoE 架构的稀疏专家层

### 多模态处理器

模型使用自定义的处理器链处理输入：
1. 文本通过 tokenizer 编码
2. 图像通过 Siglip2 编码为视觉特征
3. 视觉特征通过 LightProjector 投影到 LLM 维度
4. 构建包含图像 token 占位符的 prompt

### 权重加载

支持复杂的权重映射，包括 MoE 层的 stacked parameters 处理和 Siglip2 视觉编码器的权重加载。

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `HunyuanImage3ForConditionalGeneration` | 类 | 顶层模型类 |
| `HunyuanImage3ProcessingInfo` | 类 | 多模态处理信息 |
| `HunyuanImage3MultiModalProcessor` | 类 | 多模态处理器 |

## 与其他模块的关系

- 使用 `siglip2.py` 中的视觉编码器
- 使用 `autoencoder_kl_3d.py` 中的 3D VAE
- 继承 vLLM 的 `HunYuanModel` MoE 架构
- 实现 `SupportsMultiModal`、`SupportsMRoPE`、`SupportsPP` 接口

## 总结

HunyuanImage3 是一个大型多模态图像生成模型，通过 MoE 架构实现高效的参数利用，通过 3D VAE 实现高质量的图像重建，通过 Siglip2 实现强大的视觉理解能力。
