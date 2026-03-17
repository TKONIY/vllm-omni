# protocol/ 子模块概述

## 模块简介

`protocol/` 子模块定义了 OpenAI 兼容 API 的所有请求和响应数据模型。这些 Pydantic 模型既用于 FastAPI 的自动请求验证和文档生成，也作为服务层的数据传输对象。

## 架构图

```
protocol/
  ├── __init__.py          # 统一导出
  ├── audio.py             # TTS 相关: 语音合成请求/响应
  ├── chat_completion.py   # 聊天补全: Omni 扩展响应
  ├── images.py            # 图像生成: DALL-E 兼容协议
  └── videos.py            # 视频生成: 视频专用协议
```

## 覆盖的 API 模态

| 模态 | 请求模型 | 响应模型 |
|------|----------|----------|
| 聊天 | (继承 vLLM) | `OmniChatCompletionResponse`, `OmniChatCompletionStreamResponse` |
| 语音 | `OpenAICreateSpeechRequest`, `StreamingSpeechSessionConfig` | `AudioResponse` |
| 图像 | `ImageGenerationRequest` | `ImageGenerationResponse` |
| 视频 | `VideoGenerationRequest` | `VideoGenerationResponse`, `VideoResponse` |

## 文件索引

- [\_\_init\_\_.py](./\_\_init\_\_.py.md) — 导出汇总
- [audio.py](./audio.py.md) — 音频/TTS 协议
- [chat\_completion.py](./chat\_completion.py.md) — 聊天补全扩展
- [images.py](./images.py.md) — 图像生成协议
- [videos.py](./videos.py.md) — 视频生成协议
