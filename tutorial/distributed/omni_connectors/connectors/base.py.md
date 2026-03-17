# `base.py` — OmniConnector 抽象基类

## 文件概述

该文件定义了所有 OmniConnector 实现必须遵守的抽象基类 `OmniConnectorBase`。它规定了统一的 `put/get/cleanup/health/close` 接口，并提供了默认的序列化、key 生成和资源管理协议。

## 关键代码解析

### 抽象接口

```python
class OmniConnectorBase(ABC):
    supports_raw_data: bool = False  # RDMA 连接器可设为 True

    @abstractmethod
    def put(self, from_stage, to_stage, put_key, data) -> tuple[bool, int, dict | None]:
        """存储 Python 对象。返回 (成功, 字节数, 元数据)"""

    @abstractmethod
    def get(self, from_stage, to_stage, get_key, metadata=None) -> tuple[Any, int] | None:
        """检索 Python 对象。返回 (对象, 字节数) 或 None"""

    @abstractmethod
    def cleanup(self, request_id) -> None:
        """清理指定请求的资源"""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """返回健康状态和指标"""

    @abstractmethod
    def close(self) -> None:
        """释放资源，必须幂等"""
```

### 资源管理协议

```python
def __del__(self):
    self.close()

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
```

所有子类免费获得上下文管理器和析构器支持，只需实现 `close()`。

### 默认序列化

```python
@staticmethod
def serialize_obj(obj: Any) -> bytes:
    from ..utils.serialization import OmniSerializer
    return OmniSerializer.serialize(obj)

@staticmethod
def deserialize_obj(data: bytes) -> Any:
    from ..utils.serialization import OmniSerializer
    return OmniSerializer.deserialize(data)
```

使用集中的 `OmniSerializer`（基于 msgpack），子类可直接调用。

### Key 生成

```python
@staticmethod
def _make_key(key, from_stage, to_stage, separator="@") -> str:
    return f"{key}{separator}{from_stage}_{to_stage}"
```

默认格式：`{key}@{from_stage}_{to_stage}`。不同连接器可以重写此方法。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniConnectorBase` | ABC | 所有连接器的抽象基类 |
| `put()` | abstractmethod | 存储数据 |
| `get()` | abstractmethod | 检索数据 |
| `cleanup()` | abstractmethod | 清理资源 |
| `health()` | abstractmethod | 健康检查 |
| `close()` | abstractmethod | 关闭连接器 |
| `serialize_obj()` / `deserialize_obj()` | staticmethod | 默认序列化/反序列化 |
| `_make_key()` | staticmethod | 生成带阶段路由信息的内部 key |

## 与其他模块的关系

- 被所有具体连接器实现继承
- `serialize_obj` / `deserialize_obj` 委托给 `utils/serialization.py` 中的 `OmniSerializer`
- 被 `factory.py` 和 `adapter.py` 引用用于类型约束

## 总结

`OmniConnectorBase` 定义了连接器的标准契约：统一的数据存取接口、资源管理协议和默认工具方法。所有传输后端实现都必须继承此类并实现五个抽象方法。
