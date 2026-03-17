# `__init__.py` — distributed 模块入口

## 文件概述

此文件是 `distributed/` 包的入口，仅包含从 `omni_connectors` 子模块的批量导入，将核心 API 统一暴露给外部使用者。

## 关键代码解析

```python
from .omni_connectors import (
    ConnectorSpec,
    MooncakeConnector,
    MooncakeStoreConnector,
    MooncakeTransferEngineConnector,
    OmniConnectorBase,
    OmniConnectorFactory,
    OmniTransferConfig,
    SharedMemoryConnector,
    YuanrongConnector,
    load_omni_transfer_config,
)
```

该文件通过 `__all__` 列表控制公开 API，将配置类、连接器类、工厂类与工具函数归类导出。

## 核心类/函数

| 名称 | 用途 |
|------|------|
| `ConnectorSpec` | 连接器规格数据类 |
| `OmniTransferConfig` | 传输配置顶层数据类 |
| `OmniConnectorBase` | 连接器抽象基类 |
| `OmniConnectorFactory` | 连接器工厂 |
| `MooncakeStoreConnector` | 基于 Mooncake Store 的连接器 |
| `MooncakeTransferEngineConnector` | 基于 Mooncake RDMA 引擎的连接器 |
| `SharedMemoryConnector` | 基于共享内存的连接器 |
| `YuanrongConnector` | 基于远容 KV 存储的连接器 |
| `MooncakeConnector` | `MooncakeStoreConnector` 的向后兼容别名 |
| `load_omni_transfer_config` | 从配置文件加载传输配置 |

## 与其他模块的关系

- 作为 `distributed` 包的门面，所有实际逻辑都在 `omni_connectors` 子模块中实现。
- 外部使用者（如 `entrypoints`、`engine` 模块）通过此文件导入分布式传输相关类。

## 总结

该文件是纯导入文件，将 `omni_connectors` 的核心 API 提升到 `distributed` 包级别，方便外部模块统一引用。
