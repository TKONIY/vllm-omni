# connectors/ — 连接器实现模块

## 模块概述

`connectors/` 子模块包含 OmniConnector 的抽象基类和四种具体传输后端实现。每种实现适用于不同的部署场景。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 空初始化文件 |
| [`base.py`](base.py.md) | `OmniConnectorBase` 抽象基类 |
| [`mooncake_store_connector.py`](mooncake_store_connector.py.md) | 基于 Mooncake 分布式存储的连接器（TCP） |
| [`mooncake_transfer_engine_connector.py`](mooncake_transfer_engine_connector.py.md) | 基于 Mooncake Transfer Engine 的连接器（RDMA/零拷贝） |
| [`shm_connector.py`](shm_connector.py.md) | 基于 POSIX 共享内存的连接器（单机） |
| [`yuanrong_connector.py`](yuanrong_connector.py.md) | 基于远容 KV 客户端的连接器 |

## 连接器对比

| 特性 | MooncakeStore | MooncakeTransferEngine | SharedMemory | Yuanrong |
|------|:---:|:---:|:---:|:---:|
| 协议 | TCP | RDMA/TCP | 本地 SHM | TCP |
| 跨机器 | 是 | 是 | 否 | 是 |
| 零拷贝 | 否 | 是 | 否 | 否 |
| 依赖 | mooncake | mooncake + zmq + msgspec | 无 | datasystem |
| `supports_raw_data` | False | True | False | False |
