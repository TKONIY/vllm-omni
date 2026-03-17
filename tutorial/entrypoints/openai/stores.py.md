# `stores.py` — 异步内存存储

## 文件概述

提供了两个异步安全的内存数据结构：`TaskRegistry` 用于跟踪后台异步任务，`AsyncDictStore` 用于存储 Pydantic 模型对象。主要服务于视频生成的异步任务管理。

## 关键代码解析

### 任务注册表

```python
class TaskRegistry:
    """异步安全的后台任务注册表"""
    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, task: asyncio.Task):
        def _cleanup(_):
            asyncio.create_task(self.pop(key))  # 任务完成时自动清理
        task.add_done_callback(_cleanup)
        async with self._lock:
            self._tasks[key] = task
```

任务完成时通过 `done_callback` 自动从注册表中移除。

### 异步字典存储

```python
class AsyncDictStore(Generic[T]):
    """泛型异步安全的内存 KV 存储"""
    async def upsert(self, key, value):
        async with self._lock:
            self._items[key] = value

    async def update_fields(self, key, updates):
        """部分字段更新（使用 Pydantic 的 model_copy）"""
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            new_item = item.model_copy(update=updates)
            self._items[key] = new_item
            return new_item
```

### 全局实例

```python
VIDEO_STORE: AsyncDictStore[VideoResponse] = AsyncDictStore()
VIDEO_TASKS: TaskRegistry = TaskRegistry()
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `TaskRegistry` | 类 | 后台任务注册和自动清理 |
| `AsyncDictStore[T]` | 泛型类 | 异步安全的 KV 存储 |
| `VIDEO_STORE` | 全局实例 | 视频生成任务状态存储 |
| `VIDEO_TASKS` | 全局实例 | 视频生成后台任务注册表 |

## 与其他模块的关系

- 被 `api_server.py` 的视频生成端点使用
- `VIDEO_STORE` 存储 `VideoResponse` 对象（`protocol/videos.py`）

## 总结

两个轻量的异步安全数据结构，为视频生成的异步任务管理提供基础设施，通过 `asyncio.Lock` 保证并发安全。
