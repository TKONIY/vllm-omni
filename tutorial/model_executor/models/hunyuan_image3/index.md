# HunyuanImage3 模型模块架构概览

## 模块简介

HunyuanImage3 是腾讯混元的第三代图像生成模型，采用 HunYuan MoE LLM + Siglip2 视觉编码器 + 3D VAE 的架构。支持文本到图像生成，具有高分辨率和高质量输出能力。

## 架构图

```
文本/图像输入
       │
       ▼
┌──────────────────────────────────┐
│  HunyuanImage3 处理器             │
│  ├── Siglip2VisionTransformer    │  ← 图像视觉编码
│  │   ├── Siglip2VisionEmbeddings │
│  │   ├── Siglip2Encoder          │  ← 27 层注意力
│  │   └── Siglip2MultiheadPool   │
│  └── LightProjector             │  ← MLP 投影到 LLM 维度
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  HunyuanImage3ForConditional     │
│  Generation                      │
│  ├── HunYuanModel (MoE)         │  ← MoE LLM backbone
│  └── AutoencoderKLConv3D        │  ← 3D KL-VAE 编解码
│      ├── Encoder (3D Conv)       │
│      └── Decoder (3D Conv)       │
└──────────────────────────────────┘
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 导出 `HunyuanImage3ForConditionalGeneration` |
| `autoencoder_kl_3d.py` | 3D KL-VAE（编码器 + 解码器 + tiling） |
| `hunyuan_image3.py` | 顶层模型（HunYuan MoE + 多模态处理器） |
| `siglip2.py` | Siglip2 视觉编码器 + 投影器 |

## 核心设计思想

1. **3D VAE**：使用 3D 卷积处理时空维度，支持视频帧的联合编解码。配备空间和时间 tiling 策略，解决高分辨率显存限制。

2. **分布式解码**：VAE 解码支持多 GPU all_gather 并行，rank 0 负责 tile 融合。

3. **Siglip2 视觉编码**：使用动态分辨率位置嵌入（双线性插值），支持不同尺寸的输入图像。

4. **MoE 语言模型**：使用 HunYuan 的共享 MoE（SharedFusedMoE）架构，实现高效的文本理解和图像 token 生成。
