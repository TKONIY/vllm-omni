# `serving_speech_stream.py` — 流式 TTS WebSocket 处理器

## 文件概述

`OmniStreamingSpeechHandler` 实现了基于 WebSocket 的流式文本输入 TTS。客户端可以增量发送文本，服务器在句子边界自动分割并逐句生成音频返回。这种模式特别适用于 STT (语音转文字) -> TTS (文字转语音) 的实时语音翻译管线。

## 关键代码解析

### WebSocket 协议

```
Client -> Server:
    {"type": "session.config", ...}       # 首条消息：会话配置
    {"type": "input.text", "text": "..."}  # 增量文本块
    {"type": "input.done"}                 # 输入结束

Server -> Client:
    {"type": "audio.start", "sentence_index": 0, ...}  # 音频开始
    <binary frame: audio bytes>                          # 音频数据
    {"type": "audio.done", "sentence_index": 0}         # 音频结束
    {"type": "session.done", "total_sentences": N}      # 会话结束
    {"type": "error", "message": "..."}                  # 错误
```

### 会话处理流程

```python
async def handle_session(self, websocket: WebSocket):
    await websocket.accept()

    # 1. 等待 session.config
    config = await self._receive_config(websocket)

    # 2. 初始化句子分割器
    boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
    splitter = SentenceSplitter(boundary_re=boundary_re)

    # 3. 接收文本并逐句生成音频
    while True:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=...)
        msg = json.loads(raw)

        if msg_type == "input.text":
            sentences = splitter.add_text(msg.get("text", ""))
            for sentence in sentences:
                await self._generate_and_send(websocket, config, sentence, sentence_index)
                sentence_index += 1

        elif msg_type == "input.done":
            remaining = splitter.flush()
            if remaining:
                await self._generate_and_send(websocket, config, remaining, sentence_index)
            await websocket.send_json({"type": "session.done", "total_sentences": ...})
            return
```

### 每句音频生成

```python
async def _generate_and_send(self, websocket, config, sentence_text, sentence_index):
    # 发送 audio.start
    await websocket.send_json({"type": "audio.start", ...})

    if config.stream_audio:
        # 流式 PCM 模式：逐块发送
        async for chunk in self._speech_service._generate_pcm_chunks(generator, request_id):
            await websocket.send_bytes(chunk)
    else:
        # 完整音频模式：一次性发送
        audio_bytes, _ = await self._speech_service._generate_audio_bytes(request)
        await websocket.send_bytes(audio_bytes)

    # 发送 audio.done
    await websocket.send_json({"type": "audio.done", ...})
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniStreamingSpeechHandler` | 类 | WebSocket 流式 TTS 会话管理 |
| `handle_session()` | 异步方法 | 单个 WebSocket 会话的主循环 |
| `_receive_config()` | 异步方法 | 接收并验证会话配置 |
| `_generate_and_send()` | 异步方法 | 生成单句音频并发送 |
| `_send_error()` | 静态方法 | 发送错误消息 |

## 与其他模块的关系

- 复用 `OmniOpenAIServingSpeech`（`serving_speech.py`）的音频生成管线
- 使用 `SentenceSplitter`（`text_splitter.py`）进行句子分割
- 使用 `protocol/audio.py` 的 `StreamingSpeechSessionConfig` 配置模型
- 被 `api_server.py` 的 WebSocket 端点调用

## 总结

`OmniStreamingSpeechHandler` 将增量文本输入与逐句 TTS 结合，实现了低延迟的流式语音合成。通过句子分割器自动检测语句边界，每完成一句就立即生成并返回音频，适用于实时对话和语音翻译场景。
