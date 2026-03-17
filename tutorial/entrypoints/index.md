# entrypoints/ 模块概述

## 模块简介

`entrypoints/` 是 vLLM-Omni 项目的入口模块，负责将用户请求路由到多阶段推理管线。该模块是用户与底层引擎之间的桥梁，提供同步/异步推理入口、CLI 命令行工具、以及 OpenAI 兼容的 HTTP API 服务。

## 架构图

```
                        ┌─────────────────────────────────────┐
                        │           用户 / 客户端              │
                        └──────────┬──────────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
    ┌──────────▼──────┐  ┌─────────▼────────┐  ┌──────▼──────────┐
    │   CLI (cli/)     │  │ OpenAI API       │  │  Python SDK     │
    │  vllm serve      │  │ (openai/)        │  │  Omni/AsyncOmni │
    │  vllm bench      │  │ FastAPI 路由     │  │  直接调用        │
    └──────────┬──────┘  └─────────┬────────┘  └──────┬──────────┘
               │                   │                   │
               └───────────────────┼───────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       OmniBase (omni_base)   │
                    │   共享运行时基础 / 引擎管理   │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
   ┌──────────▼──────────┐              ┌───────────────▼──────────┐
   │    AsyncOmni         │              │   AsyncOmniDiffusion     │
   │  多阶段 LLM 管线    │              │   扩散模型推理入口        │
   │  (异步生成器模式)    │              │   (图像/视频生成)         │
   └──────────┬──────────┘              └───────────────┬──────────┘
              │                                         │
   ┌──────────▼──────────┐              ┌───────────────▼──────────┐
   │  AsyncOmniEngine     │              │   DiffusionEngine        │
   │  编排器 + 阶段引擎   │              │   扩散模型引擎            │
   └─────────────────────┘              └──────────────────────────┘
```

## 子模块说明

| 子模块 | 文件数 | 功能描述 |
|--------|--------|----------|
| **根目录** | 11 | 核心引擎入口 (Omni, AsyncOmni, AsyncOmniDiffusion) 及工具函数 |
| **cli/** | 4 | 命令行工具，`vllm serve --omni` 启动服务 |
| **cli/benchmark/** | 4 | 基准测试子命令 |
| **openai/** | 14 | OpenAI 兼容的 HTTP API 实现 |
| **openai/protocol/** | 5 | API 请求/响应的 Pydantic 数据模型 |

## 核心入口类

- **`AsyncOmni`**: 异步多阶段管线编排器，实现 `EngineClient` 接口，是在线服务的主要入口
- **`Omni`**: 同步离线推理入口，适用于批量处理
- **`AsyncOmniDiffusion`**: 异步扩散模型推理入口，用于图像生成
- **`OmniBase`**: 上述类的共享基类，封装引擎初始化和请求状态管理

## 文件索引

- [\_\_init\_\_.py](./\_\_init\_\_.py.md) — 模块入口
- [async\_omni.py](./async\_omni.py.md) — 异步多阶段管线编排器
- [async\_omni\_diffusion.py](./async\_omni\_diffusion.py.md) — 异步扩散模型入口
- [cfg\_companion\_tracker.py](./cfg\_companion\_tracker.py.md) — CFG 伴随请求跟踪器
- [chat\_utils.py](./chat\_utils.py.md) — 聊天工具函数
- [client\_request\_state.py](./client\_request\_state.py.md) — 客户端请求状态
- [omni\_base.py](./omni\_base.py.md) — 共享运行时基础
- [omni.py](./omni.py.md) — 同步推理入口
- [pd\_utils.py](./pd\_utils.py.md) — Prefill-Decode 分离工具
- [stage\_utils.py](./stage\_utils.py.md) — 阶段管理工具
- [utils.py](./utils.py.md) — 通用工具函数
- [cli/](./cli/index.md) — CLI 子模块
- [openai/](./openai/index.md) — OpenAI API 子模块
