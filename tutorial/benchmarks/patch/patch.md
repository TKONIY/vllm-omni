# `patch.py` — 猴子补丁与自定义后端注册

## 文件概述

该文件是 benchmarks 模块中最核心、最复杂的文件。它通过猴子补丁（monkey-patching）方式扩展 vLLM 的基准测试框架，实现了：

1. 注册两个新的请求后端：`openai-chat-omni` 和 `openai-audio-speech`
2. 替换 vLLM 的数据采样函数 `get_samples`，支持多模态数据集
3. 替换 vLLM 的 `benchmark` 函数，集成多模态指标计算

## 关键代码解析

### 数据采样补丁

```python
get_samples_old = datasets.get_samples

def get_samples(args, tokenizer):
    if args.backend not in ["openai-chat-omni", "openai-audio-speech"]:
        return get_samples_old(args, tokenizer)
    elif args.dataset_name == "random-mm":
        dataset = OmniRandomMultiModalDataset(random_seed=args.seed, ...)
        input_requests = dataset.sample(...)
        return input_requests
    else:
        return get_samples_old(args, tokenizer)

datasets.get_samples = get_samples  # 替换原函数
```

当后端为 omni 类型且数据集为 `random-mm` 时，使用 `OmniRandomMultiModalDataset` 生成多模态测试样本；否则回退到 vLLM 原有逻辑。

### 混合输出数据类

```python
@dataclass
class MixRequestFuncOutput(RequestFuncOutput):
    audio_ttfp: float = 0.0       # 音频首包时间
    audio_duration: float = 0.0   # 音频时长（秒）
    audio_frames: int = 0         # 音频帧数
    audio_rtf: float = 0.0        # 实时因子
    text_latency: float = 0.0     # 文本延迟
```

扩展 vLLM 的请求输出，增加音频相关字段。

### Chat Omni 请求函数

```python
async def async_request_openai_chat_omni_completions(
    request_func_input, session, pbar=None, mm_position="last"
) -> MixRequestFuncOutput:
```

该函数向 `/v1/chat/completions` 发送流式请求，关键逻辑：
- 解析 SSE 流中的 `modality` 字段区分文本和音频响应
- 文本模态：记录 TTFT 和 ITL
- 音频模态：记录 TTFP，用 pydub 解码音频块并拼接
- 最终计算 `audio_duration`、`audio_frames`、`audio_rtf`

```python
if modality == "text":
    if ttft == 0.0:
        ttft = timestamp - st
        output.ttft = ttft
    # ...
elif modality == "audio":
    if output.audio_ttfp == 0.0:
        output.audio_ttfp = timestamp - st
    audio_bytes = base64.b64decode(content)
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    generated_audio = generated_audio + seg
```

### Audio Speech 请求函数

```python
async def async_request_openai_audio_speech(
    request_func_input, session, pbar=None
) -> MixRequestFuncOutput:
```

向 `/v1/audio/speech` 发送流式 TTS 请求：
- 使用 PCM 格式（16-bit, 24kHz, mono）
- 直接按字节统计，无需解码
- 计算 `audio_duration = total_samples / sample_rate`

### 后端注册

```python
ASYNC_REQUEST_FUNCS["openai-chat-omni"] = async_request_openai_chat_omni_completions
ASYNC_REQUEST_FUNCS["openai-audio-speech"] = async_request_openai_audio_speech
```

### benchmark 函数替换

```python
async def benchmark(task_type, endpoint_type, api_url, ...) -> dict:
```

替换后的 `benchmark` 函数完整复刻了 vLLM 的基准测试流程（测试连接、预热、并发请求、结果收集），关键差异：
- 生成任务使用 `calculate_metrics`（多模态版本）
- 结果字典增加音频相关字段（`total_audio_duration_s`、`audio_throughput` 等）

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `MixRequestFuncOutput` | 数据类 | 扩展 `RequestFuncOutput`，增加音频指标字段 |
| `get_samples(args, tokenizer)` | 函数 | 替换 vLLM 的数据采样，支持多模态 |
| `async_request_openai_chat_omni_completions(...)` | 异步函数 | Chat Omni 后端请求实现 |
| `async_request_openai_audio_speech(...)` | 异步函数 | Audio Speech 后端请求实现 |
| `benchmark(...)` | 异步函数 | 替换 vLLM 的基准测试主循环 |

## 与其他模块的关系

- **依赖 data_modules**: 使用 `OmniRandomMultiModalDataset` 生成测试数据。
- **依赖 metrics**: 调用 `calculate_metrics` 计算多模态指标。
- **补丁 vLLM**: 替换 `vllm.benchmarks.datasets.get_samples` 和 `vllm.benchmarks.serve.benchmark`。
- **依赖外部库**: `aiohttp`（HTTP 请求）、`pydub`（音频解码）。

## 总结

`patch.py` 是整个 benchmarks 模块的核心连接件，通过猴子补丁技术将多模态能力无缝注入 vLLM 的基准测试框架。它注册了两个新后端（chat-omni 和 audio-speech），实现了流式音频响应的解析和指标采集，使 vllm-omni 能够对文本+音频混合模型进行全面的性能基准测试。
