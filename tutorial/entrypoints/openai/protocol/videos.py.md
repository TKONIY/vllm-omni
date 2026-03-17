# `videos.py` — 视频生成协议

## 文件概述

定义了 OpenAI 风格的视频生成 API 协议，是整个 protocol 子模块中最复杂的文件。除了请求/响应模型外，还包括视频参数解析、异步任务状态管理和视频元数据存储等数据结构。

## 关键代码解析

### 视频参数

```python
class VideoParams(BaseModel):
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: int | None = None

    @property
    def size(self) -> str | None:
        return f"{self.width}x{self.height}" if self.width and self.height else None
```

### 请求模型

```python
class VideoGenerationRequest(BaseModel):
    # OpenAI 标准字段
    model: str | None = None
    prompt: str
    size: SizeStr | None = None          # "WIDTHxHEIGHT"
    seconds: SecondStr | None = None     # 视频时长（秒）
    image_reference: ImageReference | None = None  # 参考图像

    # 视频特有字段
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    num_frames: int | None = None

    # 扩散控制扩展
    negative_prompt: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_2: float | None = None  # Wan2.2 双 CFG
    boundary_ratio: float | None = None    # Wan2.2 DiT 分割
    flow_shift: float | None = None        # 调度器偏移
    seed: int | None = None
    lora: dict[str, Any] | None = None

    def resolve_video_params(self) -> VideoParams:
        """合并顶层参数、video_params 和 size/seconds 字段"""
        vp = VideoParams(width=self.width, height=self.height, ...)
        if self.video_params:
            vp.width = vp.width or self.video_params.width
        if self.size:
            vp.width, vp.height = parse_size(self.size)
        if vp.num_frames is None and self.seconds is not None:
            vp.num_frames = int(self.seconds) * int(vp.fps)
        return vp
```

参数合并优先级：顶层字段 > video_params > size/seconds 推导。

### 图像引用类型

```python
class UrlImageReference(BaseModel):
    image_url: str

class FileImageReference(BaseModel):
    file_id: str

ImageReference = UrlImageReference | FileImageReference
```

支持 URL 引用和文件 ID 引用两种方式。

### 异步任务模型

```python
class VideoGenerationStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoResponse(BaseModel):
    """异步视频生成任务的存储模型"""
    id: str = Field(default_factory=lambda: f"video_gen_{uuid.uuid4().hex}")
    status: VideoGenerationStatus = VideoGenerationStatus.QUEUED
    progress: int = Field(default=0, description="0-100 进度")
    completed_at: int | None = None
    error: VideoError | None = None
    media_type: Literal["video/mp4"] = "video/mp4"
    file_name: str | None = None
    inference_time_s: float | None = None
```

### 列表和删除响应

```python
class VideoDeleteResponse(BaseModel):
    id: str
    deleted: bool

class VideoListResponse(BaseModel):
    first_id: str | None
    last_id: str | None
    has_more: bool
    data: list[VideoResponse]
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `VideoGenerationRequest` | Pydantic 模型 | 视频生成请求 |
| `VideoGenerationResponse` | Pydantic 模型 | 同步生成响应 |
| `VideoResponse` | Pydantic 模型 | 异步任务状态/元数据 |
| `VideoParams` | Pydantic 模型 | 视频参数块 |
| `VideoGenerationStatus` | 枚举 | 任务生命周期状态 |
| `ImageReference` | 联合类型 | 参考图像引用 |
| `VideoData` | Pydantic 模型 | 单个视频数据 |
| `VideoDeleteResponse` / `VideoListResponse` | Pydantic 模型 | CRUD 响应 |
| `file_extension()` | 函数 | MIME 类型到文件扩展名映射 |

## 与其他模块的关系

- 被 `serving_video.py` 和 `api_server.py` 使用
- `VideoResponse` 存储在 `stores.py` 的 `VIDEO_STORE` 中
- `ImageReference` 被 `video_api_utils.py` 的解码函数使用
- `parse_size()` 复用自 `image_api_utils.py`

## 总结

`videos.py` 是视频生成 API 的完整协议定义，涵盖同步/异步两种生成模式。其灵活的参数合并策略和完整的任务生命周期模型，使得同一个 API 能够服务于从简单文生视频到复杂的图生视频（带参考图像）等多种场景。
