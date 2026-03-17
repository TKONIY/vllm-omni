# `config.py` — 配置数据类

## 文件概述

该文件定义了 OmniConnector 系统的两个核心配置数据类：`ConnectorSpec`（单个连接器的规格）和 `OmniTransferConfig`（全局传输配置）。

## 关键代码解析

### ConnectorSpec — 连接器规格

```python
@dataclass
class ConnectorSpec:
    name: str  # 如 "MooncakeStoreConnector", "SharedMemoryConnector"
    extra: dict[str, Any] = field(default_factory=dict)  # 后端特定配置
```

`name` 对应 `OmniConnectorFactory` 注册表中的 key，`extra` 是传递给连接器构造函数的配置字典。

### OmniTransferConfig — 全局传输配置

```python
@dataclass
class OmniTransferConfig:
    connectors: dict[tuple[str, str], ConnectorSpec] = field(default_factory=dict)
    default_connector: ConnectorSpec | None = None

    def get_connector_for_edge(self, from_stage, to_stage) -> ConnectorSpec | None:
        edge_key = (from_stage, to_stage)
        return self.connectors.get(edge_key, self.default_connector)

    def has_connector_for_edge(self, from_stage, to_stage) -> bool:
        return self.get_connector_for_edge(from_stage, to_stage) is not None
```

核心设计：使用 `(from_stage, to_stage)` 元组作为 key，实现边（edge）级别的连接器配置。每条边可以使用不同的连接器类型。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `ConnectorSpec` | dataclass | 单个连接器的规格（名称 + 额外配置） |
| `OmniTransferConfig` | dataclass | 全局传输配置（边 → 连接器映射） |
| `get_connector_for_edge()` | method | 获取指定边的连接器规格 |
| `has_connector_for_edge()` | method | 检查指定边是否有连接器配置 |

## 与其他模块的关系

- 被 `factory.py` 中的 `create_connector()` 使用
- 被 `initialization.py` 中的配置加载函数使用
- 被 `kv_transfer_manager.py` 使用

## 总结

两个简洁的数据类定义了连接器系统的配置模型。`OmniTransferConfig` 的边级配置设计使得不同阶段之间可以灵活使用不同的传输后端。
