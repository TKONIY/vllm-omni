# `shm_connector.py` — 共享内存连接器

## 文件概述

该文件实现了基于 POSIX 共享内存的 `SharedMemoryConnector`，适用于单机多进程场景下的数据传输。它是最轻量的连接器实现，无需额外的网络依赖。

## 关键代码解析

### 初始化

```python
class SharedMemoryConnector(OmniConnectorBase):
    def __init__(self, config: dict[str, Any]):
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0, "gets": 0, "bytes_transferred": 0,
            "shm_writes": 0, "inline_writes": 0,
        }
```

`threshold` 参数原本用于决定是使用共享内存还是内联传输，但当前实现总是使用共享内存。

### put() — 写入共享内存

```python
def put(self, from_stage, to_stage, put_key, data):
    payload = self.serialize_obj(data)       # 序列化
    lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
    with open(lock_file, "wb+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)    # 文件锁
        meta = shm_write_bytes(payload, name=put_key)  # 写入 SHM
        fcntl.flock(lockf, fcntl.LOCK_UN)
    metadata = {"shm": meta, "size": size}
    return True, size, metadata
```

使用 `fcntl.flock` 文件锁保证并发安全，通过 `shm_write_bytes`（来自 `stage_utils`）写入共享内存段。

### get() — 读取共享内存

```python
def get(self, from_stage, to_stage, get_key, metadata=None):
    if metadata is not None:
        if "inline_bytes" in metadata:
            # 内联模式：直接反序列化
            return self.deserialize_obj(metadata["inline_bytes"]), size
        if "shm" in metadata:
            # SHM 模式：通过 handle 读取
            return self._get_data_with_lock(lock_file, shm_handle)
    # 无元数据：尝试按名称打开 SHM 段
    shm = shm_pkg.SharedMemory(name=get_key)
```

支持三种读取模式：
1. **内联字节**（`inline_bytes`）：直接从元数据中反序列化
2. **SHM handle**（`shm`）：通过共享内存 handle 读取
3. **按名称查找**：无元数据时回退到按 key 名称打开共享内存段

读取完成后自动清理锁文件。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `SharedMemoryConnector` | class | 基于 POSIX 共享内存的连接器 |
| `put()` | method | 序列化并写入共享内存 |
| `get()` | method | 从共享内存或内联数据读取 |
| `_get_data_with_lock()` | method | 加文件锁读取 SHM 数据 |
| `cleanup()` | method | 空操作（SHM 段在 get 时自动 unlink） |
| `health()` | method | 返回 threshold 和指标 |

## 与其他模块的关系

- 继承 `OmniConnectorBase`
- 使用 `stage_utils.shm_write_bytes` / `shm_read_bytes` 进行 SHM 操作
- 通过 `OmniConnectorFactory` 注册为 `"SharedMemoryConnector"`
- 是默认的自动配置连接器（`initialization.py` 中缺失边的自动回退）

## 总结

`SharedMemoryConnector` 是最简单的连接器实现，适用于单机部署。通过文件锁保证并发安全，利用 POSIX 共享内存实现跨进程零网络开销的数据传输。在 Ray 后端下，`shm_threshold_bytes` 会被设为 `sys.maxsize`，有效禁用 SHM 而使用 Ray 的内置传输。
