# `api_server.py` — API 服务器主文件

## 文件概述

这是 vLLM-Omni HTTP API 服务器的核心文件，约 2000 行代码。它基于 FastAPI 构建，提供了 OpenAI 兼容的 REST API 端点。该文件负责服务器的完整生命周期：引擎构建、应用状态初始化、路由注册和请求处理。

## 关键代码解析

### 服务器启动

```python
async def omni_run_server(args: Namespace) -> None:
    """主服务器启动入口"""
    # 1. 检测模型类型
    is_diffusion = is_diffusion_model(model)

    # 2. 构建引擎
    if is_diffusion:
        diffusion_engine = AsyncOmniDiffusion(model=model, **kwargs)
    else:
        async_omni = AsyncOmni(model=model, **kwargs)

    # 3. 构建 FastAPI 应用
    app = build_openai_app(args)

    # 4. 初始化应用状态（注册服务处理器）
    omni_init_app_state(async_omni, ..., app.state)

    # 5. 启动 HTTP 服务器
    serve_http(app, host=args.host, port=args.port)
```

### 应用状态初始化

```python
def omni_init_app_state(engine_client, model_config, state, args):
    """将服务处理器注入 FastAPI 应用状态"""
    # 聊天补全
    state.openai_serving_chat = OmniOpenAIServingChat(engine_client, ...)
    # 语音合成
    state.openai_serving_speech = OmniOpenAIServingSpeech(engine_client, ...)
    # 视频生成
    state.openai_serving_video = OmniOpenAIServingVideo(engine_client, ...)
```

### 路由注册

文件注册了大量 API 端点，包括但不限于：

```python
# 聊天补全
@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    ...

# 图像生成
@router.post("/v1/images/generations")
async def create_image(request: Request):
    ...

# 语音合成
@router.post("/v1/audio/speech")
async def create_speech(raw_request: Request):
    ...

# 视频生成
@router.post("/v1/videos/generations")
async def create_video(...):
    ...

# 流式 TTS (WebSocket)
@router.websocket("/v1/audio/speech/stream")
async def streaming_speech_ws(websocket: WebSocket):
    ...
```

### 图像生成内联处理

```python
@router.post("/v1/images/generations")
async def create_image(request: Request):
    # 解析请求参数
    gen_params = OmniDiffusionSamplingParams()
    if image_request.size:
        width, height = parse_size(image_request.size)
        gen_params.width, gen_params.height = width, height

    # 调用引擎生成
    result = await engine.generate(prompt=prompt, sampling_params=gen_params, ...)

    # 编码结果为 base64
    image_data = [ImageData(b64_json=encode_image_base64(img)) for img in images]
    return ImageGenerationResponse(created=int(time.time()), data=image_data)
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `omni_run_server()` | 异步函数 | 服务器完整启动流程 |
| `build_async_omni()` | 异步函数 | 构建 AsyncOmni 引擎实例 |
| `omni_init_app_state()` | 函数 | 初始化 FastAPI 应用状态 |
| `create_chat_completion()` | 路由处理器 | /v1/chat/completions |
| `create_image()` | 路由处理器 | /v1/images/generations |
| `create_speech()` | 路由处理器 | /v1/audio/speech |
| `create_video()` | 路由处理器 | /v1/videos/generations |
| `streaming_speech_ws()` | WebSocket 处理器 | /v1/audio/speech/stream |

## 与其他模块的关系

- 使用 `AsyncOmni` 和 `AsyncOmniDiffusion` 作为推理引擎
- 委托给 `serving_chat.py`、`serving_speech.py`、`serving_video.py` 处理具体业务逻辑
- 使用 `protocol/` 子模块的数据模型
- 复用 vLLM 上游的 `build_app`、`serve_http` 等基础设施
- 被 `cli/serve.py` 的 `OmniServeCommand.cmd()` 调用启动

## 总结

`api_server.py` 是整个 HTTP API 服务的中枢，负责将用户的 REST 请求路由到正确的处理器。它自动检测模型类型（LLM vs 扩散），构建相应的引擎，并注册全面的 API 端点覆盖文本、音频、图像和视频生成。
