# openai/ 子模块概述

## 模块简介

`openai/` 子模块实现了 OpenAI 兼容的 HTTP API 服务器，支持聊天补全（Chat Completions）、语音合成（TTS）、图像生成和视频生成等多种端点。它基于 FastAPI 构建，复用了 vLLM 上游的大部分 OpenAI API 基础设施，并在其上扩展了 Omni 多模态能力。

## 架构图

```
                     ┌───────────────────────────┐
                     │     FastAPI 应用           │
                     │     (api_server.py)        │
                     └─────────┬─────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         │                     │                      │
  ┌──────▼──────┐    ┌────────▼────────┐    ┌────────▼────────┐
  │ /v1/chat/   │    │ /v1/audio/      │    │ /v1/images/     │
  │ completions │    │ speech          │    │ generations     │
  │             │    │                 │    │                 │
  │ serving_    │    │ serving_        │    │ api_server.py   │
  │ chat.py     │    │ speech.py       │    │ (内联路由)      │
  └─────────────┘    └─────────────────┘    └─────────────────┘
         │
  ┌──────▼──────┐    ┌─────────────────┐    ┌─────────────────┐
  │ /v1/videos/ │    │ /v1/audio/      │    │ WebSocket       │
  │ generations │    │ speech/stream   │    │ /v1/audio/      │
  │             │    │                 │    │ speech/stream   │
  │ serving_    │    │ serving_speech  │    │                 │
  │ video.py    │    │ _stream.py      │    │ serving_speech  │
  └─────────────┘    └─────────────────┘    │ _stream.py      │
                                            └─────────────────┘
                     ┌─────────────────┐
                     │   protocol/     │
                     │  数据模型定义    │
                     │  (请求/响应)     │
                     └─────────────────┘
```

## API 端点

| 端点 | 方法 | 处理类 | 功能 |
|------|------|--------|------|
| `/v1/chat/completions` | POST | `OmniOpenAIServingChat` | 聊天补全（文本/音频/图像） |
| `/v1/audio/speech` | POST | `OmniOpenAIServingSpeech` | 语音合成 (TTS) |
| `/v1/audio/speech/stream` | WebSocket | `OmniStreamingSpeechHandler` | 流式 TTS |
| `/v1/images/generations` | POST | 内联 | 图像生成 |
| `/v1/videos/generations` | POST | `OmniOpenAIServingVideo` | 视频生成 |
| `/v1/audio/speech/speakers` | GET/POST/DELETE | `OmniOpenAIServingSpeech` | 说话人管理 |

## 文件索引

- [\_\_init\_\_.py](./\_\_init\_\_.py.md) — 模块入口
- [api\_server.py](./api\_server.py.md) — API 服务器主文件
- [audio\_utils\_mixin.py](./audio\_utils\_mixin.py.md) — 音频工具 Mixin
- [errors.py](./errors.py.md) — 自定义错误
- [image\_api\_utils.py](./image\_api\_utils.py.md) — 图像 API 工具
- [metadata\_manager.py](./metadata\_manager.py.md) — 元数据管理器
- [serving\_chat.py](./serving\_chat.py.md) — 聊天补全服务
- [serving\_speech.py](./serving\_speech.py.md) — 语音合成服务
- [serving\_speech\_stream.py](./serving\_speech\_stream.py.md) — 流式 TTS
- [serving\_video.py](./serving\_video.py.md) — 视频生成服务
- [storage.py](./storage.py.md) — 本地存储管理
- [stores.py](./stores.py.md) — 异步内存存储
- [text\_splitter.py](./text\_splitter.py.md) — 句子分割器
- [video\_api\_utils.py](./video\_api\_utils.py.md) — 视频 API 工具
- [protocol/](./protocol/index.md) — 协议数据模型
