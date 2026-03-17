# GLM-Image 模型模块架构概览

## 模块简介

GLM-Image 是智谱 AI 的图像生成模型，支持文本到图像（t2i）和图像到图像（i2i）生成。基于 Qwen2 LLM + VQ-VAE 架构，使用自回归方式生成图像 token，并通过 M-RoPE（多维旋转位置编码）处理图像的空间位置信息。

## 架构图

```
文本/图像输入
       │
       ▼
┌──────────────────────────────────┐
│  GlmImageMultiModalProcessor     │  ← 图像预处理 + prompt 构建
│  GlmImageProcessingInfo          │  ← 配置/处理器路径解析
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  GlmImageForConditionalGeneration│
│  ├── Vision Encoder              │  ← 图像编码（i2i 模式）
│  │   ├── Patch Embedding         │
│  │   └── Attention Blocks        │
│  ├── VQ-VAE Tokenizer            │  ← 图像特征量化
│  ├── Qwen2 LLM                   │  ← 自回归 token 生成
│  │   └── M-RoPE                  │  ← 多维旋转位置编码
│  └── compute_logits()            │  ← 图像/文本 token 输出
└──────────────────────────────────┘
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 导出 `GlmImageForConditionalGeneration` |
| `glm_image_ar.py` | 完整的 AR 模型实现（~900行） |

## 核心设计思想

1. **M-RoPE 位置编码**：使用多维 RoPE 处理图像的网格位置（temporal, height, width），使模型能够理解空间结构。

2. **双模式处理**：t2i 模式只需文本 prompt 和目标网格尺寸；i2i 模式额外需要源图像的视觉特征。

3. **特殊目录结构**：模型文件存放在 `vision_language_encoder/` 子目录，处理器在 `processor/` 子目录，ProcessingInfo 中自动解析路径。
