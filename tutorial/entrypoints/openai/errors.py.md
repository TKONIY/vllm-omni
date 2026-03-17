# `errors.py` — 自定义错误

## 文件概述

定义了 OpenAI API 层使用的自定义异常类。

## 关键代码解析

```python
class InvalidInputReferenceError(ValueError):
    def __init__(self, message: str = "Invalid input reference.") -> None:
        super().__init__(message)
```

`InvalidInputReferenceError` 继承自 `ValueError`，用于标识图像/视频输入引用无效的情况（如 base64 解码失败、URL 下载失败等）。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `InvalidInputReferenceError` | 异常类 | 输入引用无效时抛出 |

## 与其他模块的关系

- 被 `video_api_utils.py` 在图像解码失败时抛出
- 被 `api_server.py` 的路由处理器捕获并转为 HTTP 错误响应

## 总结

一个简单的自定义异常类，用于在多媒体输入验证失败时提供清晰的错误信息。
