# stable_audio/ -- Stable Audio Open 模型目录索引

## 目录概述

Stable Audio Open 是 Stability AI 的文本到音频模型，在 1D 音频潜空间中使用 DiT 进行扩散生成。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化 |
| [`stable_audio_transformer.py`](stable_audio_transformer.md) | 1D DiT 模型（GQA, SwiGLU, 部分 RoPE） |
| [`pipeline_stable_audio.py`](pipeline_stable_audio.md) | 文本到音频管线（T5, Oobleck VAE） |

## 架构概览

```
输入: 文本 prompt + 音频时长
  -> T5 编码 + 投影
  -> 时长条件编码 (start/end seconds)
  -> 准备 1D 噪声潜变量 [B, 64, L]
  -> 24 层 DiT (self-attn + cross-attn + SwiGLU FFN)
  -> Oobleck VAE 解码
输出: 音频波形
```

## 核心特色

1. **音频生成**: 1D 潜空间 (而非 2D 图像)
2. **GQA**: 交叉注意力使用 24 Q 头 / 12 KV 头
3. **时长条件**: 精确控制生成音频的起止时间
4. **部分 RoPE**: 仅 head_dim 前半部分应用旋转编码
5. **Oobleck VAE**: 专为音频设计的自编码器

## 总结

Stable Audio Open 展示了 vllm-omni 扩散框架对音频模态的支持能力，通过 `SupportAudioOutput` 接口声明音频输出能力，使引擎能正确处理音频格式的后处理。
