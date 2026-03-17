# `factory.py` — OmniConnector 工厂

## 文件概述

该文件实现了 `OmniConnectorFactory` 工厂类，基于注册表模式管理所有可用连接器的创建。文件底部注册了所有内置连接器类型。

## 关键代码解析

### 工厂核心

```python
class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""
    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}

    @classmethod
    def register_connector(cls, name: str, constructor):
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}")
        constructor = cls._registry[spec.name]
        connector = constructor(spec.extra)
        return connector
```

工厂使用类变量 `_registry` 作为全局注册表，key 是连接器名称字符串，value 是构造函数。

### 延迟导入的构造函数

每个连接器都有一个独立的构造函数，使用延迟导入避免启动时加载不必要的依赖：

```python
def _create_mooncake_store_connector(config):
    from .connectors.mooncake_store_connector import MooncakeStoreConnector
    return MooncakeStoreConnector(config)
```

### 内置注册

```python
OmniConnectorFactory.register_connector("MooncakeStoreConnector", _create_mooncake_store_connector)
OmniConnectorFactory.register_connector("MooncakeTransferEngineConnector", _create_mooncake_transfer_engine_connector)
OmniConnectorFactory.register_connector("SharedMemoryConnector", _create_shm_connector)
OmniConnectorFactory.register_connector("YuanrongConnector", _create_yuanrong_connector)
# 向后兼容别名
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_store_connector)
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniConnectorFactory` | class | 连接器工厂，管理注册与创建 |
| `register_connector()` | classmethod | 注册一个连接器构造函数 |
| `create_connector()` | classmethod | 根据 `ConnectorSpec` 创建连接器实例 |
| `list_registered_connectors()` | classmethod | 列出所有已注册的连接器名称 |

## 与其他模块的关系

- 使用 `ConnectorSpec` 数据类作为创建参数
- 被 `initialization.py` 中的 `create_connectors_from_config()` 调用
- 被 `OmniKVTransferManager` 调用以惰性创建连接器
- 被 `OmniChunkTransferAdapter` 调用创建分块传输连接器

## 总结

`factory.py` 通过工厂模式和注册表实现了连接器的可插拔创建。延迟导入设计确保只有实际使用的连接器才会加载其依赖库。向后兼容别名 `MooncakeConnector` 保证旧配置文件仍然可用。
