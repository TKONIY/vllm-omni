# `yuanrong_connector.py` — 远容 KV 存储连接器

## 文件概述

该文件实现了基于远容 (Yuanrong) Datasystem KV 客户端的 `YuanrongConnector`，适用于使用远容分布式存储系统的部署环境。

## 关键代码解析

### 依赖导入

```python
try:
    from datasystem.kv_client import KVClient, SetParam, WriteMode
except ImportError:
    KVClient = None
```

依赖 `datasystem` 包，可选安装。

### 初始化

```python
class YuanrongConnector(OmniConnectorBase):
    def __init__(self, config):
        self.host = config.get("host", "127.0.0.1")
        self.port = int(config.get("port", "35001"))
        self.client = KVClient(self.host, self.port)
        self.client.init()

        self.set_param = SetParam()
        self.set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT
        self.get_sub_timeout_ms = max(0, int(config.get("get_sub_timeout_ms", 1000)))
```

### Key 格式

```python
def _make_key(self, rid, from_stage, to_stage) -> str:
    return f"{rid}:{from_stage}_{to_stage}"
```

注意：重写了基类的 `_make_key`，使用冒号 `:` 而非 `@` 作为分隔符。

### put / get

```python
def put(self, from_stage, to_stage, put_key, data):
    serialized_data = self.serialize_obj(data)
    key = self._make_key(put_key, from_stage, to_stage)
    self.client.set(key, serialized_data, self.set_param.write_mode)
    return True, len(serialized_data), None

def get(self, from_stage, to_stage, get_key, metadata=None):
    key = self._make_key(get_key, from_stage, to_stage)
    raw_list = self.client.get([key], False, self.get_sub_timeout_ms)
    raw_data = raw_list[0] if raw_list else None
    if raw_data is not None:
        data = self.deserialize_obj(raw_data)
        return data, len(raw_data)
```

`get()` 使用可配置的超时时间（默认 1000ms）。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `YuanrongConnector` | class | 基于远容 KV 客户端的连接器 |
| `put()` | method | 序列化并存储到远容 |
| `get()` | method | 从远容检索并反序列化 |
| `_make_key()` | method | 生成 `rid:from_to` 格式的 key |
| `cleanup()` | method | 空操作 |
| `close()` | method | 将客户端置空 |

## 与其他模块的关系

- 继承 `OmniConnectorBase`
- 依赖 `datasystem` 包的 `KVClient`
- 通过 `OmniConnectorFactory` 注册为 `"YuanrongConnector"`

## 总结

`YuanrongConnector` 提供了与远容分布式存储系统的集成。实现模式与 `MooncakeStoreConnector` 类似，都是基于 KV 存储的序列化传输，区别在于底层依赖不同的存储后端。
