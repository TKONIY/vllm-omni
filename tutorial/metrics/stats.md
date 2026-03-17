# `stats.py` — 编排器指标聚合器

## 文件概述

该文件是 vllm-omni 指标系统的核心，定义了多阶段推理流水线的性能统计数据结构和 `OrchestratorAggregator` 聚合器类。聚合器跟踪每个阶段的生成时间、token 吞吐、数据传输、端到端延迟等指标，并生成格式化的日志摘要。

## 关键代码解析

### 数据结构层次

```python
@dataclass
class StageStats:
    total_token: int = 0
    total_gen_time_ms: float = 0.0

    @property
    def avg_tokens_per_s(self) -> float:
        return (self.total_token * 1000.0 / self.total_gen_time_ms) if self.total_gen_time_ms > 0 else 0.0
```

`StageStats` 是阶段级的累积统计，跟踪总 token 数和总生成时间。

```python
@dataclass
class StageRequestStats:
    batch_id: int
    batch_size: int
    num_tokens_in: int
    num_tokens_out: int
    stage_gen_time_ms: float
    rx_transfer_bytes: int        # 接收传输字节数
    rx_decode_time_ms: float      # 接收解码时间
    rx_in_flight_time_ms: float   # 数据在途时间
    stage_stats: StageStats
    audio_generated_frames: int = 0
    diffusion_metrics: dict[str, int] = None
```

`StageRequestStats` 记录单个请求在某个阶段的详细指标，包括输入/输出 token 数、生成时间、传输开销等。

```python
@dataclass
class TransferEdgeStats:
    from_stage: int
    to_stage: int
    request_id: str
    size_bytes: int
    tx_time_ms: float
    used_shm: bool = False       # 是否使用共享内存

@dataclass
class RequestE2EStats:
    request_id: str
    e2e_total_ms: float
    e2e_total_tokens: int
    transfers_total_time_ms: float
    transfers_total_bytes: int
```

`TransferEdgeStats` 跟踪阶段间数据传输，`RequestE2EStats` 记录请求的端到端统计。

### 聚合器核心

```python
class OrchestratorAggregator:
    def __init__(self, num_stages, log_stats, wall_start_ts, final_stage_id_for_e2e):
        self.stage_events: dict[str, list[StageRequestStats]] = {}
        self.transfer_events: dict[tuple[int, int, str], TransferEdgeStats] = {}
        self.e2e_events: list[RequestE2EStats] = []
```

聚合器维护三类事件存储：
- `stage_events`: 按 request_id 索引的阶段统计列表
- `transfer_events`: 按 `(from_stage, to_stage, request_id)` 索引的传输统计
- `e2e_events`: 端到端统计列表

### 核心方法

**记录阶段指标：**

```python
def process_stage_metrics(self, *, result, stage_type, stage_id, req_id,
                          engine_outputs, finished, final_output_type, output_to_yield):
    _m = result.get("metrics")
    if _m is not None:
        self.accumulated_gen_time_ms[req_id][stage_id] += _m.stage_gen_time_ms
        if finished:
            self.on_stage_metrics(stage_id, req_id, _m, final_output_type)
    if output_to_yield is not None and finished:
        self.record_audio_generated_frames(output_to_yield, stage_id, req_id)
```

**记录传输指标：**

```python
def on_forward(self, from_stage, to_stage, req_id, size_bytes, tx_ms, used_shm):
    if self.stage_first_ts[to_stage] is None:
        self.stage_first_ts[to_stage] = time.time()
    self.record_transfer_tx(from_stage, to_stage, req_id, size_bytes, tx_ms, used_shm)
```

**请求完成处理：**

```python
def on_finalize_request(self, stage_id, req_id, req_start_ts):
    e2e_ms = (time.time() - req_start_ts) * 1000.0
    # 汇总该请求所有阶段的 token 数
    total_tokens = sum(evt.num_tokens_in + evt.num_tokens_out for evt in self.stage_events[rid_key])
    self.e2e_events.append(RequestE2EStats(...))
```

**构建摘要：**

```python
def build_and_log_summary(self) -> dict[str, Any]:
    # 计算 wall time、平均请求时间、平均 token 吞吐
    # 按 request_id 遍历，生成 stage 表、transfer 表、e2e 表
    # 使用 _format_table 格式化并通过 logger.info 输出
```

### 阶段后处理计时器

```python
@contextmanager
def stage_postprocess_timer(self, stage_id, req_id):
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        _postproc_ms = (time.perf_counter() - _t0) * 1000.0
        self.record_stage_postprocess_time(stage_id, req_id, _postproc_ms)
```

提供上下文管理器，方便测量阶段后处理时间。

### 字段转换配置

```python
FIELD_TRANSFORMS = {
    "rx_transfer_bytes": ("rx_transfer_kbytes", lambda v: v / 1024.0),
    "size_bytes": ("size_kbytes", lambda v: v / 1024.0),
    "transfers_total_bytes": ("transfers_total_kbytes", lambda v: v / 1024.0),
}
STAGE_EXCLUDE = {"stage_stats", "stage_id", "request_id", ...}
```

通过配置驱动表格显示：指定字段转换（如字节转千字节）和排除不需要显示的字段。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `StageStats` | 数据类 | 阶段级累积统计（token 数、生成时间） |
| `StageRequestStats` | 数据类 | 请求在特定阶段的详细指标 |
| `TransferEdgeStats` | 数据类 | 阶段间数据传输统计 |
| `RequestE2EStats` | 数据类 | 请求端到端统计 |
| `OrchestratorAggregator` | 类 | 编排器指标聚合器，跟踪全部事件 |
| `process_stage_metrics(...)` | 方法 | 处理并记录阶段指标 |
| `on_forward(...)` | 方法 | 记录阶段间数据转发 |
| `on_finalize_request(...)` | 方法 | 请求完成时的端到端指标汇总 |
| `build_and_log_summary(...)` | 方法 | 构建并输出格式化摘要 |
| `stage_postprocess_timer(...)` | 上下文管理器 | 阶段后处理计时 |

## 与其他模块的关系

- **被编排器调用**: 本模块是编排器（orchestrator）的指标基础设施，在多阶段流水线中被调用。
- **依赖 utils**: 使用 `_build_field_defs`、`_build_row`、`_format_table` 进行表格构建和格式化。
- **与 benchmarks/metrics 不同**: benchmarks/metrics 面向外部基准测试客户端，本模块面向内部推理流水线。

## 总结

`stats.py` 实现了一个完整的多阶段推理流水线指标收集框架。它通过四个数据类（StageStats、StageRequestStats、TransferEdgeStats、RequestE2EStats）捕获从阶段生成、阶段间传输到端到端延迟的全链路性能数据，并通过 `OrchestratorAggregator` 将这些数据聚合、计算并格式化输出，为多模态模型推理的性能分析和优化提供了有力支撑。
