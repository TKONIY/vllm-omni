# `client_request_state.py` — 客户端请求状态

## 文件概述

定义了 `ClientRequestState` 类，用于在编排器中跟踪单个请求的运行时状态。每个进入 `AsyncOmni` 或 `Omni` 的请求都会创建一个对应的状态对象。

## 关键代码解析

```python
class ClientRequestState:
    """跟踪编排器中单个请求的状态"""

    def __init__(self, request_id: str, queue: asyncio.Queue | None = None):
        self.request_id = request_id
        self.stage_id: int | None = None           # 当前所在阶段
        self.queue = queue if queue is not None else asyncio.Queue()  # 输出消息队列
        self.metrics: OrchestratorAggregator | None = None  # 性能指标聚合器
```

每个字段的作用：
- `request_id`: 请求的唯一标识符
- `stage_id`: 记录请求当前在哪个阶段执行
- `queue`: 异步队列，后台输出分发任务将编排器的结果放入此队列，`generate()` 方法从中读取
- `metrics`: 每请求的性能指标聚合器，记录各阶段的延迟和吞吐

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `ClientRequestState` | 类 | 单个请求的运行时状态容器 |

## 与其他模块的关系

- 被 `AsyncOmni` 和 `Omni` 创建并存储在 `request_states` 字典中
- `OmniBase._handle_output_message()` 通过它路由输出消息
- `OmniBase._process_single_result()` 通过它访问性能指标

## 总结

一个轻量的状态容器，将请求 ID、当前阶段、输出队列和性能指标聚合在一起，是编排器实现按请求隔离的基础数据结构。
