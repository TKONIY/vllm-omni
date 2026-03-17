# omni_coordinator/ — 服务协调器模块

## 模块概述

`omni_coordinator/` 子模块实现了分布式多阶段推理的服务协调系统。它管理阶段实例的注册、状态监控、心跳检测和实例列表广播，并提供负载均衡能力。

## 架构设计

```
                    ┌─────────────────────┐
                    │   OmniCoordinator   │
                    │   (ROUTER + PUB)    │
                    └───────┬─────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
 ┌────────▼────────┐  ┌────▼────────┐  ┌──────▼───────────┐
 │ Stage Client    │  │ Stage Client│  │ Hub Client       │
 │ (DEALER)        │  │ (DEALER)    │  │ (SUB)            │
 │ 发送 heartbeat  │  │ 发送 update │  │ 接收 instance    │
 │ 和 status       │  │             │  │ list 更新        │
 └─────────────────┘  └─────────────┘  └──────────────────┘
```

- **OmniCoordinator**: 中心服务，通过 ROUTER 接收 Stage 事件，通过 PUB 广播实例列表
- **OmniCoordClientForStage**: Stage 端客户端，发送注册、状态更新和心跳
- **OmniCoordClientForHub**: Hub 端客户端（AsyncOmni 侧），订阅实例列表用于负载均衡

## 文件列表

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.py.md) | 包入口，导出所有公共接口 |
| [`messages.py`](messages.py.md) | 消息数据类：`InstanceEvent`、`InstanceInfo`、`InstanceList`、`StageStatus` |
| [`load_balancer.py`](load_balancer.py.md) | 负载均衡器：`LoadBalancer` 基类和 `RandomBalancer` 实现 |
| [`omni_coordinator.py`](omni_coordinator.py.md) | 协调器服务：`OmniCoordinator` |
| [`omni_coord_client_for_hub.py`](omni_coord_client_for_hub.py.md) | Hub 端客户端：`OmniCoordClientForHub` |
| [`omni_coord_client_for_stage.py`](omni_coord_client_for_stage.py.md) | Stage 端客户端：`OmniCoordClientForStage` |
