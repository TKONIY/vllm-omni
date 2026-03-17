# distributed/ — vllm-omni 分布式模块总览

## 模块概述

`distributed/` 是 vllm-omni 项目的分布式基础设施模块，负责多阶段（multi-stage）推理管线中的数据传输、KV 缓存转移、服务协调与负载均衡。该模块是实现 Prefill-Decode 分离（PD disaggregation）和多阶段 pipeline 推理的核心。

## 子模块结构

| 子模块 | 说明 |
|--------|------|
| [kv_transfer/](kv_transfer/index.md) | Monkey-patch vLLM 原生 KV Transfer 连接器，修复 PD 分离场景下的 request-ID 不匹配问题 |
| [omni_connectors/](omni_connectors/index.md) | OmniConnector 核心框架：连接器抽象基类、工厂、多种传输后端实现（Mooncake、共享内存、远容）、传输适配器、序列化与配置工具 |
| [omni_coordinator/](omni_coordinator/index.md) | 服务协调器：管理多阶段实例的注册、心跳、状态发布与负载均衡 |
| [ray_utils/](ray_utils/index.md) | Ray 分布式相关工具函数：集群初始化、Placement Group 管理、Actor 生命周期管理 |

## 架构关系

```
                    ┌──────────────────┐
                    │ omni_coordinator │  服务发现与负载均衡
                    └────────┬─────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                      │
   ┌────▼────┐        ┌─────▼──────┐         ┌─────▼────┐
   │ Stage-0 │───────▶│  Stage-1   │────────▶│ Stage-N  │
   │(Prefill)│        │ (Decode/..)│         │(Diffusion│
   └─────────┘        └────────────┘         │   /...)  │
        │                    │                └──────────┘
        └────────────────────┘
          通过 omni_connectors 传输数据
          (Mooncake / SHM / Yuanrong / RDMA)

   ray_utils: 可选的 Ray 后端支持
   kv_transfer: PD 分离时 KV 缓存传输的补丁
```

## `__init__.py` 说明

顶层 `__init__.py` 从 `omni_connectors` 子模块导出所有公共接口，包括配置类（`ConnectorSpec`、`OmniTransferConfig`）、连接器基类与实现类（`OmniConnectorBase`、`MooncakeStoreConnector`、`SharedMemoryConnector` 等）、工厂类（`OmniConnectorFactory`）以及工具函数（`load_omni_transfer_config`）。
