# `__init__.py` — omni_connectors 包入口

## 文件概述

该文件是 `omni_connectors` 包的入口，负责从各子模块中导入并统一导出所有公共接口。

## 关键代码解析

```python
from .connectors.base import OmniConnectorBase
from .connectors.mooncake_store_connector import MooncakeStoreConnector
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector

try:
    from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
except ImportError:
    MooncakeTransferEngineConnector = None  # RDMA deps not installed

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec, OmniTransferConfig
from .utils.initialization import (
    build_stage_connectors,
    get_connectors_config_for_stage,
    # ...
)

# Backward-compatible alias
MooncakeConnector = MooncakeStoreConnector
```

注意事项：
- `MooncakeTransferEngineConnector` 依赖 `msgspec`、`zmq`、`mooncake` 等包，使用 `try/except` 做可选导入
- `MooncakeConnector` 是 `MooncakeStoreConnector` 的向后兼容别名

## 核心类/函数

| 名称 | 用途 |
|------|------|
| `OmniConnectorBase` | 连接器抽象基类 |
| `OmniConnectorFactory` | 连接器工厂 |
| `ConnectorSpec` / `OmniTransferConfig` | 配置数据类 |
| 四种连接器实现 | Mooncake Store / RDMA / SHM / Yuanrong |
| `load_omni_transfer_config` | 从文件加载配置 |
| `initialize_connectors_from_config` | 从配置初始化所有连接器 |
| `build_stage_connectors` | 为特定 Stage 构建连接器 |

## 与其他模块的关系

- 被上层 `distributed/__init__.py` 再次导出
- 所有具体连接器实现在 `connectors/` 子模块中
- 配置和工具在 `utils/` 子模块中

## 总结

统一导出文件，聚合了连接器框架的所有公共 API。
