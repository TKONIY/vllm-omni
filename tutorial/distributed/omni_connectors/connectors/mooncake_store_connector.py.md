# `mooncake_store_connector.py` — Mooncake 分布式存储连接器

## 文件概述

该文件实现了基于 Mooncake 分布式存储的 `MooncakeStoreConnector`。Mooncake 是一个分布式 KV 存储系统，该连接器通过 TCP 协议进行数据传输，适用于跨机器的阶段间数据交换。

## 关键代码解析

### 初始化

```python
class MooncakeStoreConnector(OmniConnectorBase):
    def __init__(self, config: dict[str, Any]):
        self.host = config.get("host", "127.0.0.1")
        self.metadata = config.get("metadata_server", "http://127.0.0.1:8080/metadata")
        self.master = config.get("master", "127.0.0.1:50051")
        self.segment = config.get("segment", 512 * 1024 * 1024)   # 512MB
        self.localbuf = config.get("localbuf", 64 * 1024 * 1024)   # 64MB
        self.proto = config.get("proto", "tcp")
        self.rdma = config.get("rdma", "")
        self._init_store()
```

配置参数包括 Mooncake 的 metadata 服务器、master 地址、段大小、本地缓冲区大小和传输协议。

### 数据存储 (put)

```python
def put(self, from_stage, to_stage, put_key, data):
    serialized_data = self.serialize_obj(data)          # 使用 OmniSerializer
    key = self._make_key(put_key, from_stage, to_stage) # 生成带路由的 key
    self.store.put(key, serialized_data, self.pin)      # 写入 Mooncake 存储
    return True, len(serialized_data), None
```

### 数据检索 (get) — 带重试

```python
def get(self, from_stage, to_stage, get_key, metadata=None):
    retries = 20
    sleep_s = 0.05
    key = self._make_key(get_key, from_stage, to_stage)
    for attempt in range(retries):
        raw_data = self.store.get(key)
        if raw_data:
            data = self.deserialize_obj(raw_data)
            return data, len(raw_data)
        time.sleep(sleep_s)
    return None
```

最多重试 20 次，每次间隔 50ms（总超时约 1 秒）。记录详细的性能指标（fetch 时间、反序列化时间、吞吐量）。

### 清理和关闭

- `cleanup()`: 空操作（Mooncake 不支持显式删除，靠 GC 回收）
- `close()`: 调用 `self.store.close()` 关闭连接

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `MooncakeStoreConnector` | class | 基于 Mooncake Store 的连接器 |
| `_init_store()` | method | 初始化 Mooncake 存储和 ReplicateConfig |
| `put()` | method | 序列化并存储数据 |
| `get()` | method | 带重试地检索和反序列化数据 |
| `health()` | method | 返回连接和指标信息 |

## 与其他模块的关系

- 继承 `OmniConnectorBase`，使用其 `serialize_obj/deserialize_obj` 和 `_make_key`
- 依赖 `mooncake` 包（`MooncakeDistributedStore`、`ReplicateConfig`）
- 通过 `OmniConnectorFactory` 注册为 `"MooncakeStoreConnector"` 和 `"MooncakeConnector"`（兼容别名）

## 总结

`MooncakeStoreConnector` 是最基础的跨机器连接器实现。它利用 Mooncake 分布式存储的 KV 接口完成数据传输，适合不需要零拷贝或 RDMA 的场景。内置的重试机制和详细日志使其在生产环境中具有较好的可观测性。
