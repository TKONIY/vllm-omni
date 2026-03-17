# `chat_completion.py` — 聊天补全扩展

## 文件概述

在 vLLM 标准的 `ChatCompletionResponse` 和 `ChatCompletionStreamResponse` 基础上，添加 Omni 特有的扩展字段。

## 关键代码解析

```python
class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"         # 当前输出模态（text/audio/image）
    metrics: dict[str, Any] | None = None  # 多阶段性能指标

class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None  # 多阶段性能指标
```

扩展字段说明：
- `modality`: 标识当前流式 chunk 的输出模态，客户端可据此决定如何渲染
- `metrics`: 包含各阶段延迟、吞吐等性能指标，用于调试和监控

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniChatCompletionStreamResponse` | Pydantic 模型 | 流式响应（增加 modality 和 metrics） |
| `OmniChatCompletionResponse` | Pydantic 模型 | 完整响应（增加 metrics） |

## 与其他模块的关系

- 被 `serving_chat.py` 用于构造多模态聊天响应
- 继承 vLLM 的标准聊天补全协议，保持 API 兼容性

## 总结

两个轻量的子类扩展，为 OpenAI 聊天补全协议添加了多模态标识和性能指标字段，实现了 Omni 特有功能与标准协议的无缝融合。
