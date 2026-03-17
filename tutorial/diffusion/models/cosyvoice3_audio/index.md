# cosyvoice3_audio/ -- CosyVoice3 音频模型目录索引

## 目录概述

CosyVoice3 是一个语音合成扩散模型，使用 DiT 架构在 Mel 频谱域进行去噪生成。

## 文件列表

| 文件 | 说明 | 文档 |
|------|------|------|
| [`__init__.py`](__init__.md) | 包初始化（空） |
| [`cosyvoice3_dit.py`](cosyvoice3_dit.md) | DiT 模型（自注意力 + 多模态输入融合） |

## 核心特色

1. **多模态输入融合**: 噪声音频 + 条件音频 + 文本嵌入 + 说话人嵌入
2. **因果卷积位置编码**: 支持流式推理
3. **优化注意力后端**: 通过 vllm-omni 使用 FlashAttention/SageAttention/SDPA
4. **可选长跳跃连接**: 类 U-Net 残差路径

## 总结

CosyVoice3 DiT 专为语音合成设计，通过因果卷积和多条件融合机制实现高质量、可流式的语音生成。
