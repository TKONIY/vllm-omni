# `serving_video.py` — 视频生成服务

## 文件概述

`OmniOpenAIServingVideo` 实现了 OpenAI 风格的视频生成 API，提供 `/v1/videos/generations` 端点。支持文生视频（text-to-video）和图生视频（image-to-video）模式，以及视频音频混合输出。

## 关键代码解析

### 视频生成流程

```python
async def generate_videos(self, request, reference_id, *, reference_image=None):
    # 1. 构建提示（支持参考图像和负面提示）
    prompt: OmniTextPrompt = OmniTextPrompt(prompt=request.prompt)
    if reference_image is not None:
        prompt["multi_modal_data"] = {"image": input_image}

    # 2. 设置生成参数
    gen_params = OmniDiffusionSamplingParams()
    vp = request.resolve_video_params()
    gen_params.width, gen_params.height = vp.width, vp.height
    gen_params.num_frames = vp.num_frames
    gen_params.fps = vp.fps

    # 3. 执行生成
    result = await self._run_generation(prompt, gen_params, reference_id)

    # 4. 提取视频和音频输出
    videos = self._extract_video_outputs(result)
    audios = self._extract_audio_outputs(result, expected_count=len(videos))

    # 5. 编码为 Base64 MP4
    video_data = [VideoData(b64_json=encode_video_base64(video, fps=fps, audio=audio, ...))]
    return VideoGenerationResponse(created=int(time.time()), data=video_data)
```

### 视频输出归一化

```python
@staticmethod
def _normalize_video_outputs(videos):
    """将各种格式的视频输出统一为列表"""
    if hasattr(videos, "ndim") and videos.ndim == 5:
        return [videos[i] for i in range(videos.shape[0])]  # batch 拆分
    if isinstance(videos, list):
        # 处理嵌套列表、PIL 图像列表等多种格式
```

### LoRA 支持

```python
@staticmethod
def _apply_lora(lora_body, gen_params):
    """从请求中提取 LoRA 配置并注入采样参数"""
    lora_name = lora_body.get("name") or lora_body.get("lora_name")
    lora_path = lora_body.get("local_path") or lora_body.get("path")
    lora_int_id = stable_lora_int_id(str(lora_path))  # 稳定的整数 ID
    gen_params.lora_request = LoRARequest(str(lora_name), int(lora_int_id), str(lora_path))
```

### 音频采样率解析

```python
def _resolve_audio_sample_rate(self, result):
    """多层回退策略解析音频采样率"""
    # 1. 尝试从生成结果中获取
    # 2. 尝试从模型配置中获取
    # 3. 默认 24000 Hz
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniOpenAIServingVideo` | 类 | 视频生成请求处理器 |
| `generate_videos()` | 异步方法 | 执行视频生成并返回响应 |
| `for_diffusion()` | 类方法 | 创建扩散模式实例 |
| `_run_generation()` | 异步方法 | 调用引擎执行生成 |
| `_extract_video_outputs()` | 方法 | 从结果提取视频 |
| `_extract_audio_outputs()` | 静态方法 | 从结果提取音频 |
| `_apply_lora()` | 静态方法 | 处理 LoRA 请求 |
| `_resolve_audio_sample_rate()` | 方法 | 解析音频采样率 |
| `ReferenceImage` | 数据类 | 参考图像包装 |

## 与其他模块的关系

- 使用 `AsyncOmni` 引擎（多阶段模式）或 `AsyncOmniDiffusion`（纯扩散模式）
- 使用 `video_api_utils.py` 编码视频为 Base64 MP4
- 使用 `protocol/videos.py` 的请求/响应数据模型
- 被 `api_server.py` 注册为视频生成端点的处理器

## 总结

`OmniOpenAIServingVideo` 提供了完整的视频生成 API，支持文生视频、图生视频、LoRA 适配和视频音频混合输出。它通过灵活的输出格式归一化和多层采样率回退策略，兼容多种不同的视频扩散模型输出格式。
