# `image_api_utils.py` — 图像 API 工具

## 文件概述

为 OpenAI 兼容的图像生成 API 提供工具函数，包括尺寸字符串解析和图像 Base64 编码。

## 关键代码解析

```python
def parse_size(size_str: str) -> tuple[int, int]:
    """解析 'WIDTHxHEIGHT' 格式的尺寸字符串"""
    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(f"无效尺寸格式: '{size_str}'，期望 'WIDTHxHEIGHT'")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"宽高必须为正整数")
    return width, height

def encode_image_base64(image: PIL.Image.Image) -> str:
    """将 PIL 图像编码为 Base64 PNG 字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `parse_size()` | 函数 | 解析 "1024x1024" 格式尺寸 |
| `encode_image_base64()` | 函数 | PIL 图像转 Base64 |

## 与其他模块的关系

- 被 `api_server.py` 的图像生成端点使用
- 被 `serving_chat.py` 在多阶段图像生成中使用
- `parse_size()` 也被 `protocol/videos.py` 使用

## 总结

两个简洁的工具函数，分别处理尺寸解析和图像编码，是图像生成 API 的基础构件。
