# engine/ 模块概览 — vLLM-Omni 多阶段推理引擎

## 模块简介

`engine/` 模块是 vLLM-Omni 项目的核心引擎层，负责管理多阶段（multi-stage）推理流水线的完整生命周期。它在 vLLM 原生引擎之上构建了一套编排（Orchestrator）机制，使得多个模型阶段（如 thinker、talker、diffusion）能够协同工作，实现文本、语音、图像等多模态推理任务。

## 架构总览

```
                          用户请求 (PromptType)
                               |
                               v
                    +---------------------+
                    | AsyncOmniEngine     |  <-- 用户线程中的轻量代理
                    | (async_omni_engine) |
                    +---------------------+
                         |           ^
                   janus queue    janus queue
                   (request)      (output)
                         |           |
                         v           |
                    +---------------------+
                    |   Orchestrator       |  <-- 后台线程 asyncio 事件循环
                    |   (orchestrator.py)  |
                    +---------------------+
                    /        |         \
                   v         v          v
           +----------+ +----------+ +----------+
           | Stage 0  | | Stage 1  | | Stage N  |
           | (LLM/    | | (LLM/    | | (LLM/    |
           | Diffusion)| | Diffusion)| | Diffusion)|
           +----------+ +----------+ +----------+
               |              |             |
               v              v             v
        StageEngine     StageEngine    StageDiffusion
        CoreClient      CoreClient     Client
        (vLLM AsyncMP)  (vLLM AsyncMP)
```

## 文件索引

| 文件 | 功能描述 |
|------|----------|
| [`__init__.py`](__init__.md) | 定义核心数据结构：`OmniEngineCoreRequest`、`OmniEngineCoreOutput` 等 msgspec 序列化载体 |
| [`arg_utils.py`](arg_utils.md) | 引擎参数类 `OmniEngineArgs` / `AsyncOmniEngineArgs`，扩展 vLLM 原生参数以支持多阶段配置 |
| [`async_omni_engine.py`](async_omni_engine.md) | `AsyncOmniEngine` — 面向用户的异步引擎代理，管理 Orchestrator 生命周期 |
| [`orchestrator.py`](orchestrator.md) | `Orchestrator` — 后台线程的核心调度器，负责阶段间请求转发与输出路由 |
| [`output_processor.py`](output_processor.md) | `MultimodalOutputProcessor` — 多模态输出处理器，支持张量累积与合并 |
| [`serialization.py`](serialization.md) | 序列化工具函数，处理 `additional_information` 的张量/列表编解码 |
| [`stage_engine_core_client.py`](stage_engine_core_client.md) | `StageEngineCoreClient` — 单阶段异步客户端，继承 vLLM `AsyncMPClient` |
| [`stage_init.py`](stage_init.md) | 阶段初始化辅助函数：设备映射、引擎参数构建、设备锁管理 |
| [`worker_cls_utils.py`](worker_cls_utils.md) | Worker 类解析工具，根据 `worker_type` 选择 AR 或 Generation Worker |

## 数据流说明

1. **请求入口**：用户通过 `AsyncOmniEngine.add_request()` 提交请求
2. **输入处理**：Stage-0 的 `InputProcessor` 在调用者线程中完成 tokenization 和多模态预处理
3. **队列传输**：处理后的 `OmniEngineCoreRequest` 通过 janus 队列发送到 Orchestrator 线程
4. **阶段执行**：Orchestrator 将请求提交到对应的 `StageEngineCoreClient`
5. **输出处理**：`MultimodalOutputProcessor` 处理原始输出，累积多模态张量
6. **阶段转发**：如果存在下游阶段，Orchestrator 通过 `_forward_to_next_stage()` 将输出转发
7. **结果返回**：最终输出通过 output janus 队列返回给用户

## 关键设计模式

- **线程隔离**：`AsyncOmniEngine`（用户线程）与 `Orchestrator`（后台线程）通过 janus 队列通信，避免 GIL 竞争
- **阶段抽象**：LLM 阶段和 Diffusion 阶段使用统一的 `StageMetadata` 描述，支持灵活组合
- **Async-Chunk 模式**：通过共享内存连接器实现阶段间流式数据传输
- **msgspec 序列化**：使用高性能 msgspec 进行 ZMQ 跨进程数据传输
