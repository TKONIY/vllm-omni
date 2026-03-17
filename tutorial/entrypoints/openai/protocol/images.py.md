# `images.py` — 图像生成协议

## 文件概述

定义了 OpenAI DALL-E 兼容的图像生成 API 协议，包括请求、响应和响应格式枚举。在标准 DALL-E 参数基础上扩展了扩散模型特有的控制参数。

## 关键代码解析

### 请求模型

```python
class ImageGenerationRequest(BaseModel):
    # OpenAI 标准字段
    prompt: str                          # 文本描述
    model: str | None = None             # 模型名称
    n: int = Field(default=1, ge=1, le=10)  # 生成数量
    size: str | None = None              # 尺寸 "WIDTHxHEIGHT"
    response_format: ResponseFormat = ResponseFormat.B64_JSON

    # vllm-omni 扩散控制扩展
    negative_prompt: str | None = None    # 负面提示
    num_inference_steps: int | None = None  # 推理步数 (1-200)
    guidance_scale: float | None = None   # CFG 引导系数 (0-20)
    true_cfg_scale: float | None = None   # True CFG 系数
    seed: int | None = None               # 随机种子
    generator_device: str | None = None   # 生成器设备

    # LoRA 支持
    lora: dict[str, Any] | None = None    # LoRA 适配器配置

    # VAE 优化
    vae_use_slicing: bool | None = False
    vae_use_tiling: bool | None = False
```

### 响应模型

```python
class ImageData(BaseModel):
    b64_json: str | None = None    # Base64 编码的 PNG 图像
    url: str | None = None         # 图像 URL（未实现）
    revised_prompt: str | None = None

class ImageGenerationResponse(BaseModel):
    created: int                   # Unix 时间戳
    data: list[ImageData]          # 生成的图像列表
    output_format: str = None      # 输出格式
    size: str = None               # 实际尺寸
```

### 验证器

```python
@field_validator("response_format")
def validate_response_format(cls, v):
    if v is not None and v != ResponseFormat.B64_JSON:
        raise ValueError("仅支持 'b64_json' 格式")
    return v
```

当前仅支持 Base64 JSON 格式返回。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `ResponseFormat` | 枚举 | 响应格式（b64_json / url） |
| `ImageGenerationRequest` | Pydantic 模型 | 图像生成请求 |
| `ImageData` | Pydantic 模型 | 单张图像数据 |
| `ImageGenerationResponse` | Pydantic 模型 | 图像生成响应 |

## 与其他模块的关系

- 被 `api_server.py` 的 `/v1/images/generations` 端点使用
- 在 `protocol/__init__.py` 中统一导出

## 总结

标准的 DALL-E API 兼容协议，在保持 OpenAI 接口规范的同时，扩展了扩散模型特有的控制参数（推理步数、CFG 引导、LoRA 等），为高级用户提供了细粒度的生成控制。
