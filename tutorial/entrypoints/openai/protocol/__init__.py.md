# `__init__.py` — 协议导出汇总

## 文件概述

统一导出 protocol 子模块中的核心数据模型，方便上层模块导入。

## 关键代码解析

```python
from vllm_omni.entrypoints.openai.protocol.chat_completion import OmniChatCompletionStreamResponse
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData, ImageGenerationRequest, ImageGenerationResponse, ResponseFormat,
)
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoData, VideoGenerationRequest, VideoGenerationResponse,
)
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `ImageGenerationRequest` / `ImageGenerationResponse` | 类 | 图像生成协议 |
| `VideoGenerationRequest` / `VideoGenerationResponse` | 类 | 视频生成协议 |
| `OmniChatCompletionStreamResponse` | 类 | 流式聊天响应扩展 |
| `ResponseFormat` | 枚举 | 图像响应格式 |

## 总结

标准的包导出文件，将分散在各文件中的协议模型统一到一个导入路径下。
