# omni_connectors/ — OmniConnector 核心框架

## 模块概述

`omni_connectors/` 是 vllm-omni 分布式传输的核心模块，提供了一套完整的数据传输框架，用于多阶段推理管线中各 Stage 之间的数据交换。该模块涵盖连接器抽象、多种传输后端、工厂模式、传输适配器、序列化工具等。

## 子模块与文件

| 路径 | 说明 |
|------|------|
| [connectors/](connectors/index.md) | 连接器实现：基类 + 4 种后端（Mooncake Store、Mooncake RDMA、共享内存、远容） |
| [transfer_adapter/](transfer_adapter/index.md) | 传输适配器：管理异步分块传输的调度逻辑 |
| [utils/](utils/index.md) | 工具模块：配置、序列化、日志、初始化、KV 工具 |
| [`__init__.py`](__init__.py.md) | 包入口，统一导出所有公共接口 |
| [`adapter.py`](adapter.py.md) | Orchestrator 级别的发送/接收适配函数 |
| [`factory.py`](factory.py.md) | 连接器工厂：注册和创建连接器实例 |
| [`kv_transfer_manager.py`](kv_transfer_manager.py.md) | KV 缓存传输管理器：提取、发送和接收 KV 缓存 |

## 架构设计

```
OmniConnectorFactory (工厂)
    │
    ├── MooncakeStoreConnector     (TCP/分布式存储)
    ├── MooncakeTransferEngineConnector (RDMA/零拷贝)
    ├── SharedMemoryConnector      (本机共享内存)
    └── YuanrongConnector          (远容 KV 客户端)
          │
          └── 都继承自 OmniConnectorBase

adapter.py: try_send_via_connector / try_recv_via_connector
    └── 调用连接器的 put() / get()

OmniKVTransferManager
    └── 管理 KV 缓存从 GPU 提取 → 通过连接器传输 → 接收并还原到 GPU

OmniChunkTransferAdapter
    └── 管理分块 (chunk) 级别的异步传输与调度集成
```

## 总结

`omni_connectors/` 是整个分布式传输的中枢，通过工厂模式支持可插拔的传输后端，通过适配器层屏蔽底层差异，为上层推理引擎提供统一的数据传输接口。
