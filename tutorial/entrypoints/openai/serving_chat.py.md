# `serving_chat.py` — 聊天补全服务

## 文件概述

`OmniOpenAIServingChat` 是 `/v1/chat/completions` 端点的核心处理类。它继承自 vLLM 的 `OpenAIServingChat` 并混入 `AudioMixin`，在标准聊天补全基础上扩展了多模态输出（音频、图像）和扩散模型图像生成能力。

## 关键代码解析

### 双模式设计

```python
class OmniOpenAIServingChat(OpenAIServingChat, AudioMixin):
    _diffusion_mode: bool = False
    _diffusion_engine: Optional["AsyncOmniDiffusion"] = None

    @classmethod
    def for_diffusion(cls, diffusion_engine, model_name):
        """创建扩散模式实例"""
        instance = cls.__new__(cls)
        instance._diffusion_mode = True
        instance._diffusion_engine = diffusion_engine
        return instance
```

通过 `for_diffusion()` 工厂方法可以创建扩散模式实例，在同一个聊天接口下支持图像生成。

### 多阶段图像生成

```python
async def create_chat_completion(self, request, raw_request):
    if self._diffusion_mode:
        return await self._create_diffusion_chat_completion(request, raw_request)

    # 多模态图像生成：检测 modalities 中是否包含 "image"
    if request.modalities and ("image" in request.modalities):
        # 从聊天消息中提取文本提示和参考图像
        extracted_prompt, reference_images = self._extract_diffusion_prompt_and_images(messages)
        # 构建 OmniTextPrompt 而非预分词的 prompt
        tprompt: OmniTextPrompt = {"prompt": extracted_prompt}
        engine_prompts = [tprompt]
```

当请求包含图像模态时，跳过标准的 chat template 预处理，直接将原始文本和参考图像传给多阶段管线。

### 提示预处理

```python
async def _preprocess_chat(self, request, messages, ...):
    # 音频-视频联合理解：注入视频中的音频
    if mm_proc_kw.get("use_audio_in_video", False):
        messages = await self._inject_audio_from_video_urls(messages)

    # 使用 vLLM 的 renderer 系统处理消息
    (conversation,), (engine_prompt,) = await renderer.render_chat_async(
        [messages], chat_params, tok_params, ...
    )
```

### 流式与非流式输出

```python
if request.stream:
    return self.chat_completion_stream_generator(
        request, result_generator, request_id, model_name, ...
    )
else:
    return await self.chat_completion_full_generator(
        request, result_generator, request_id, model_name, ...
    )
```

支持标准的 SSE 流式响应和完整响应两种模式。在流式模式下，多阶段管线的输出会逐步流式返回。

### 采样参数构建

```python
def _build_sampling_params_list_from_request(self, request):
    """从 OpenAI API 请求参数构建多阶段采样参数"""
    # 理解阶段使用标准 SamplingParams
    # 扩散阶段使用 OmniDiffusionSamplingParams
    # TTS 阶段使用 OmniSamplingParams
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniOpenAIServingChat` | 类 | 聊天补全请求处理器 |
| `for_diffusion()` | 类方法 | 创建扩散模式实例 |
| `create_chat_completion()` | 异步方法 | 处理聊天补全请求 |
| `_preprocess_chat()` | 异步方法 | 聊天消息预处理 |
| `chat_completion_stream_generator()` | 异步生成器 | 流式输出 |
| `chat_completion_full_generator()` | 异步方法 | 非流式输出 |
| `_inject_audio_from_video_urls()` | 异步方法 | 视频音频注入 |
| `_extract_diffusion_prompt_and_images()` | 方法 | 从消息提取图像生成提示 |

## 与其他模块的关系

- 继承 vLLM 的 `OpenAIServingChat` 获取标准聊天补全能力
- 混入 `AudioMixin`（`audio_utils_mixin.py`）获取音频处理能力
- 使用 `AsyncOmni` 作为 LLM 引擎客户端
- 使用 `AsyncOmniDiffusion` 作为扩散引擎（扩散模式）
- 使用 `chat_utils.py` 提取视频音频
- 输出使用 `protocol/chat_completion.py` 的数据模型

## 总结

`OmniOpenAIServingChat` 是 vLLM-Omni API 层最复杂的组件，将标准 LLM 聊天补全扩展为支持多模态输出（文本+音频+图像）的统一接口。它的双模式设计（LLM 管线 vs 纯扩散）和自动提示路由（chat template vs 原始文本）使同一个 API 端点能够服务于各种多模态模型。
