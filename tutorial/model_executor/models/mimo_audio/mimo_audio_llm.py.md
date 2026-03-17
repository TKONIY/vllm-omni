# `mimo_audio_llm.py` — MiMo-Audio LLM 模型

## 文件概述

MiMo-Audio 的核心 LLM 模型实现，基于 Qwen2 扩展了多通道音频嵌入、局部 Transformer、以及 Qwen2Audio 风格的多模态处理。该文件约 1500 行，是 MiMo-Audio 最复杂的组件。

## 关键代码解析

该模型的核心架构包括：

1. **Qwen2 backbone**：使用 vLLM 的 Qwen2Model + PagedAttention 作为主干 LLM
2. **多通道语音嵌入**：8 个通道各自拥有独立的 Embedding 表
3. **输入局部 Transformer**：处理音频输入特征的局部上下文
4. **输出局部 Transformer**：预测下一步的多通道 codec token
5. **维度投影**：在 LLM 隐空间和局部 Transformer 之间进行维度转换

### 多模态嵌入

模型支持 Qwen2Audio 风格的音频处理：
- 使用 `Qwen2AudioProcessingInfo` 兼容的处理流程
- 音频特征通过 Whisper 编码器提取
- 在 `embed_multimodal` 中注入音频嵌入

### 权重加载

支持复杂的权重映射，包括：
- Qwen2 标准的 stacked parameters（q/k/v → qkv_proj）
- 多通道语音嵌入的逐通道加载
- 局部 Transformer 的权重映射

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `MiMoAudioLLMModel` | 类 | 完整的 LLM + 音频模型 |
| `embed_multimodal()` | 方法 | 音频多模态嵌入注入 |
| `embed_input_ids()` | 方法 | 文本 + 音频 token 嵌入 |
| `load_weights()` | 方法 | 多组件权重加载 |

## 与其他模块的关系

- 使用 vLLM 的 `Qwen2Model` 作为 backbone
- 使用 `Qwen2AudioProcessingInfo` 的音频处理兼容接口
- 被 `mimo_audio.py` 的 `fused_thinker_talker` 阶段实例化

## 总结

MiMo-Audio LLM 的核心创新在于融合 Thinker-Talker 架构——在同一 Qwen2 模型中同时完成文本理解（Thinker）和多通道音频 token 预测（Talker），通过局部 Transformer 处理通道间的依赖关系。
