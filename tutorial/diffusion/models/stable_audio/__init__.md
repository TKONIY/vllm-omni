# `stable_audio/__init__.py` -- Stable Audio 模型包初始化

## 文件概述

Stable Audio Open 扩散模型子包入口，导出 DiT 模型、Pipeline 和后处理函数。

**文件路径**: `vllm_omni/diffusion/models/stable_audio/__init__.py`

## 导出内容

| 名称 | 类型 | 说明 |
|------|------|------|
| `StableAudioDiTModel` | Transformer 类 | 音频 DiT 模型 |
| `StableAudioPipeline` | Pipeline 类 | 文本到音频管线 |
| `get_stable_audio_post_process_func` | 工厂函数 | 音频后处理 |

## 总结

Stable Audio Open 是 Stability AI 的文本到音频模型，使用 DiT 架构在音频潜空间中进行去噪生成。
