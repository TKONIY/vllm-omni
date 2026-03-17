# `video_api_utils.py` — 视频 API 工具

## 文件概述

为视频生成 API 提供全套工具函数，覆盖图像引用解码、视频张量归一化、MP4 编码和 Base64 编码等功能。支持 PyTorch 张量、NumPy 数组和 PIL 图像等多种输入格式。

## 关键代码解析

### 图像引用解码

```python
async def decode_input_reference(image_reference, input_reference_bytes):
    """支持多种输入方式的图像解码"""
    if isinstance(input_reference_bytes, bytes):
        return _decode_image_bytes(input_reference_bytes, source="input_reference")
    if isinstance(image_reference, UrlImageReference):
        return await decode_image_url(image_reference.image_url)
    elif isinstance(image_reference, FileImageReference):
        raise InvalidInputReferenceError("file_id 暂不支持")
    return None

async def decode_image_url(image_url):
    """支持 data URL 和 HTTP URL"""
    if image_url.startswith("data:image"):
        return _decode_base64_image(image_url, ...)
    if image_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(image_url)
        return _decode_image_bytes(response.content, ...)
```

### 视频张量归一化

```python
def _normalize_video_tensor(video_tensor: torch.Tensor) -> np.ndarray:
    """将 PyTorch 视频张量归一化为 (F, H, W, C) numpy 数组"""
    video_tensor = video_tensor.detach().cpu()
    if video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
        video_tensor = video_tensor.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]
    if video_tensor.is_floating_point():
        video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]
    return video_tensor.numpy()
```

自动处理多种通道排列（CFHW、FCHW、FHWC）和数值范围（[-1,1]、[0,1]、[0,255]）。

### MP4 编码（带可选音频）

```python
def _encode_video_bytes(video, fps, audio=None, audio_sample_rate=None):
    frames = _coerce_video_to_frames(video)
    if audio is not None:
        # 使用 LTX2 的编码器混合音视频
        from diffusers.pipelines.ltx2.export_utils import encode_video
        encode_video(video_tensor, fps=fps, audio=waveform, ...)
    else:
        # 纯视频导出
        export_to_video(frames, tmp_file.name, fps=fps)
```

### 音频波形预处理

```python
def _coerce_audio_to_waveform(audio):
    """将各种音频格式统一为 2 声道 float tensor"""
    waveform = waveform.squeeze()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # 单声道
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # 复制为立体声
    return waveform.float().contiguous()
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `decode_input_reference()` | 异步函数 | 解码图像输入引用 |
| `decode_image_url()` | 异步函数 | 下载/解码图像 URL |
| `_normalize_video_tensor()` | 函数 | PyTorch 视频张量归一化 |
| `_normalize_video_array()` | 函数 | NumPy 视频数组归一化 |
| `_coerce_video_to_frames()` | 函数 | 将各种视频格式转为帧列表 |
| `_coerce_audio_to_waveform()` | 函数 | 音频波形预处理 |
| `_encode_video_bytes()` | 函数 | 编码为 MP4 字节 |
| `encode_video_base64()` | 函数 | 编码为 Base64 MP4 |

## 与其他模块的关系

- 被 `serving_video.py` 的 `generate_videos()` 使用
- 被 `api_server.py` 的视频相关端点使用
- 使用 `errors.py` 的 `InvalidInputReferenceError`
- 使用 `protocol/videos.py` 的 `ImageReference` 类型
- 依赖 `diffusers` 的 `export_to_video` 和 `encode_video`

## 总结

`video_api_utils.py` 是视频 API 的数据处理引擎，负责将各种格式的模型输出（张量、数组、帧列表）统一归一化并编码为标准 MP4。其鲁棒的格式检测和多路径归一化策略确保了对不同视频扩散模型输出的广泛兼容。
