# `storage.py` — 本地存储管理

## 文件概述

提供了异步安全的本地文件存储管理器，用于保存和管理 API 服务产生的文件（如说话人音频样本、生成的视频等）。

## 关键代码解析

```python
class LocalStorageManager(StorageBaseManager):
    def __init__(self, storage_path=None):
        if storage_path is None:
            storage_path = os.getenv("VLLM_OMNI_STORAGE_PATH", "/tmp/storage")
        self.storage_path = storage_path
        max_concurrency = int(os.getenv("VLLM_OMNI_STORAGE_MAX_CONCURRENCY", "4"))
        self._io_semaphore = asyncio.Semaphore(max(1, max_concurrency))

    def _save_sync(self, data: bytes, file_name: str) -> str:
        """原子性文件保存：临时文件 + replace"""
        with NamedTemporaryFile("wb", dir=self.storage_path, delete=False) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # 确保数据落盘
        os.replace(tmp_name, filename)  # 原子替换

    async def save(self, data, file_name) -> str:
        async with self._io_semaphore:
            return await asyncio.to_thread(self._save_sync, data, file_name)
```

关键设计：
1. 通过 `Semaphore` 控制并发 I/O 数量（默认 4）
2. 使用临时文件 + `os.replace` 实现原子写入
3. 调用 `os.fsync` 确保数据持久化
4. 所有 I/O 操作通过 `asyncio.to_thread` 异步化

```python
# 全局单例
STORAGE_MANAGER = LocalStorageManager()
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `StorageBaseManager` | 抽象类 | 存储管理器接口 |
| `LocalStorageManager` | 类 | 本地文件系统存储实现 |
| `save()` | 异步方法 | 保存文件 |
| `delete()` | 异步方法 | 删除文件 |
| `exists()` | 方法 | 检查文件是否存在 |
| `STORAGE_MANAGER` | 全局实例 | 默认存储管理器 |

## 与其他模块的关系

- 被 `serving_speech.py` 用于保存上传的说话人音频
- 被 `api_server.py` 用于保存生成的视频文件
- 存储路径可通过环境变量 `VLLM_OMNI_STORAGE_PATH` 配置

## 总结

一个轻量但健壮的本地文件存储管理器，通过并发控制、原子写入和异步 I/O 确保在高并发 API 服务中的数据安全。
