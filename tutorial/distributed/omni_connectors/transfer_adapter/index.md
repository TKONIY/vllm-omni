# transfer_adapter/ — 传输适配器模块

## 模块概述

`transfer_adapter/` 子模块提供传输适配器（Transfer Adapter），用于管理分块（chunk）级别的异步数据传输。适配器封装了连接器的使用细节，并与调度器（Scheduler）集成，处理请求在等待数据和就绪之间的状态转换。

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 空初始化文件 |
| [`base.py`](base.py.md) | `OmniTransferAdapterBase` 基类：后台收发循环和请求队列 |
| [`chunk_transfer_adapter.py`](chunk_transfer_adapter.py.md) | `OmniChunkTransferAdapter`：分块传输适配器，集成调度器状态管理 |

## 架构设计

```
OmniChunkTransferAdapter
    ├── recv_loop (后台线程) ──── _poll_single_request() ──── connector.get()
    ├── save_loop (后台线程) ──── _send_single_request() ──── connector.put()
    ├── load_async()    ← Scheduler 调用
    ├── save_async()    ← Model Runner 调用
    └── process_pending_chunks()  ← Scheduler 调度前调用
         └── WAITING_FOR_CHUNK ↔ WAITING/RUNNING 状态转换
```
