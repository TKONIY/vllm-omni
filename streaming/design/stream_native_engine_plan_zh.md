# Stream-Native Engine 计划

状态：research prototype 讨论稿
Owner：Yangshen
日期：2026-05-18

本文讨论如何为 Qwen-Omni、Qwen-TTS 以及 Thinking Machines Lab interactive workloads
快速构建一个 stream-native research prototype。目标不是兼容 vLLM-Omni 的所有历史路径，
也不是继续给 `async_chunk` 补特殊分支，而是旁路旧 orchestrator，新写一个窄用途、
chunk-native、deadline-native 的研究 engine。

核心原则：

- 旧 orchestrator 不需要立刻删除，但 prototype 不走它的 request progression。
- 新 engine 只服务少数固定 graph：Qwen3-TTS、Qwen3-Omni、interactive session。
- 先证明 scheduling / trigger / credit / session runtime 的研究结论，再考虑接回生产路径。
- 允许牺牲通用性、平台兼容性和完整 OpenAI API 兼容，换实现速度和实验可控性。

## 当前状态

当前系统已经有不少可复用的传输和模型执行基础，但控制平面仍然是 request-native：

- PR [#727](https://github.com/vllm-project/vllm-omni/pull/727) 增加了
  Qwen3-Omni 跨 stage async chunk，并报告单请求 E2E 从约 3869 ms 降到约
  1305 ms。对应 RFC [#268](https://github.com/vllm-project/vllm-omni/issues/268)
  明确把跨 stage prefill pipeline 和 async put/get 留作后续工作。
- RFC [#3509](https://github.com/vllm-project/vllm-omni/issues/3509) 关注
  async D2H/H2D、pinned pool 和低拷贝 SHM wire path。它是传输层 hardening，
  不是调度策略重构。
- RFC [#3535](https://github.com/vllm-project/vllm-omni/issues/3535) 展示了
  用户可感知的 SLO 缺口：Qwen3-TTS Base voice_clone 在 concurrency
  64/128/256 时 underrun p99 约为 16.2/17.7/19.1 s，而目标是 100 ms。
  该 RFC 明确把 scheduler/SLA 工作列为 out of scope。
- PR [#3485](https://github.com/vllm-project/vllm-omni/pull/3485) 修复了
  TTS 延迟回退，并加入 first-chunk-only 的 `initial_codec_chunk_frames`，但
  trigger 仍然只是局部启发式。
- PR [#3322](https://github.com/vllm-project/vllm-omni/pull/3322) 恢复了
  Qwen3-TTS Code2Wav cross-request batching。这是重要吞吐优化，但 active stream
  选择和 underrun deadline 仍然需要新的 scheduler。

本地关键代码路径：

- `vllm_omni/engine/orchestrator.py` 负责 request progression。普通模式下 stage
  完成后才 forward；`async_chunk` 模式下用 placeholder prompt length 预热下游
  stage。
- `vllm_omni/core/sched/omni_ar_scheduler.py` 和
  `vllm_omni/core/sched/omni_generation_scheduler.py` 仍然是 vLLM request scheduler，
  只是加了 Omni hook。
- `vllm_omni/core/sched/omni_scheduling_coordinator.py`、
  `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`
  和 `vllm_omni/worker/omni_connector_model_runner_mixin.py` 管理
  `WAITING_FOR_CHUNK`、chunk ready signal、chunk id 和 connector I/O。
- `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py` 的 dynamic first
  chunk 只看 active request 数和 stage capacity。`qwen3_omni.py` 的 Talker 到
  Code2Wav 仍然使用固定 `codec_chunk_frames`。
- `vllm_omni/entrypoints/openai/api_server.py` 在 `async_chunk` 开启时直接拒绝
  `/v1/realtime`。

## 为什么需要新 Engine

缺失的抽象不是更快的 connector，而是 stream graph。一个 stream chunk 应该同时带有
upstream producer、downstream credit、已累积 payload size、session/request 身份、
由 SLO 和播放状态导出的 deadline，以及 completion/merge policy。今天这些信息分散在
模型特定 processor、scheduler status mutation、queue polling 和 frontend 代码里。

这种拆分造成四个结构性限制：

- 调度是 stage-local 且 SLO-unaware。下游 Code2Wav 看到的是 ready request，而不是
  带 deadline 或 playback debt 的 chunk。
- emission policy 嵌在 processor 里。`initial_codec_chunk_frames=1` 可以改善 TTFA，
  但看不到 downstream queue depth、用户 underrun 或 overload 状态。
- prefill 在跨 stage 上仍然是 barrier。已有 intra-stage chunked prefill，但
  encoder/Thinker/Talker 的依赖没有表示为可增量下发的 downstream work。
- bounded request 和 unbounded session 是两套 serving mode。API 层目前阻止
  `async_chunk` 和 `/v1/realtime` 共存。

## Research Prototype 范围

这个版本不是 production replacement，而是一个旁路研究 engine：

```text
vLLM-Omni old path:
  OpenAI API -> AsyncOmniEngine -> Orchestrator -> StagePool -> EngineCore

research path:
  research entrypoint -> ResearchStreamEngine -> StreamScheduler
    -> StageActor -> existing EngineCore / model runner
```

旧 orchestrator 先保留，作为生产实现和对照 baseline。新路径直接创建固定 graph、管理
stream event、调度 stage actor。这样可以抛弃 `WAITING_FOR_CHUNK`、placeholder prewarm、
full_payload/async_chunk 双路径兼容和 server-level API 互斥。

第一版只支持：

- **Qwen3-TTS**：`text/session input -> Talker -> Code2Wav -> audio chunks`
- **Qwen3-Omni**：`text/audio/video input -> Thinker -> Talker -> Code2Wav -> text/audio chunks`
- **interactive session**：`session input chunks -> Thinker state -> response stream`

明确不支持：

- arbitrary model pipeline
- diffusion / PD / CFG
- NPU/XPU/多平台兼容
- 完整 OpenAI API 行为兼容
- legacy YAML 全字段兼容
- 多租户生产级 fault isolation

## 目标 Runtime

stream-native engine 应该把调度单元定义为 typed stream event：

```text
StreamEvent {
  stream_id, session_id, request_id,
  stage_id, edge_id, seq_no,
  kind: prefill | decode | codec | waveform | control,
  payload_ref,
  ready_time, deadline, slo_class,
  upstream_progress, downstream_credit,
  size_hint, cost_hint,
  policy_state
}
```

runtime 包含五个核心组件：

- **Stream graph registry**：stage actor 和 edge 组成的 DAG。用显式
  producer/consumer dependency 替代硬编码 `stage_id + 1` progression。
- **Trigger policy engine**：每条 edge 上根据 deadline、downstream credit、
  upstream progress、accumulated size 和质量约束做 `emit | hold | merge` 决策。
- **Global chunk scheduler**：对 ready stream event 做 EDF 或 weighted EDF，同时考虑
  每个 stage 的资源约束、SLO class、admission control 和 overload degradation。
- **Credit/backpressure manager**：管理 per-edge 和 per-stream credit。credit 由 consumer
  拥有，producer emit 前必须消耗 credit。
- **统一 session/request lifecycle**：bounded TTS/chat request 和 unbounded realtime/VLA
  session 都是 stream，只是结束条件不同。

## 从 Flink / Spark 借鉴什么

Flink 和 Spark 对我们最有价值的不是 API surface，而是 runtime 分层方式：用户逻辑被编译成
dataflow graph；每个节点有清晰 lifecycle、state、timer、metrics；边上有 backpressure；
系统用 trigger/watermark/checkpoint 表达时间、进度和一致性。LLM inference engine 应该借鉴
这些控制面抽象，但不要照搬 SQL/DataFrame、exactly-once sink 或通用窗口语义。

### 抽象映射

| Flink / Spark 概念 | LLM streaming inference 中的对应物 | 第一版怎么做 |
| --- | --- | --- |
| Dataflow graph / logical plan | Qwen3-TTS、Qwen3-Omni、interactive 的固定 stream graph | 先 hardcode preset，不做通用 graph compiler |
| Operator / StreamTask | `StageActor` 或轻量 policy/pack operator | `TalkerActor`、`Code2WavActor`、`PackerOperator` |
| Operator chain | 同进程、同 GPU 上可同步执行的轻量链 | 只 chain pack/policy，不把重 GPU stage 过早 fuse |
| Keyed state | per-session / per-stream state | key 用 `session_id`、`stream_id`，状态包括 KV handle、playback buffer、codec history |
| Watermark / event time | input progress、token position、audio playback time、deadline watermark | 用于 deadline 和过期判断，不用于 SQL window finalization |
| Trigger | `emit | hold | merge` policy | 从 fixed chunk size 改为 deadline+credit+upstream progress |
| Backpressure | bounded edge queue + consumer credit | producer 没 credit 不 emit |
| Checkpoint barrier | coherent session snapshot / trace barrier | research 阶段先做 trace barrier，生产再做恢复语义 |
| Spark micro-batch trigger | GPU batch builder 的 flush 条件 | 由 deadline、max batch size、GPU slot、credit 共同触发 |
| Output mode | append / update / complete stream semantics | audio waveform 用 append；speculative text/session revise 用 update；final result 用 complete |

### 为什么不能直接照搬

- Flink/Spark 的主问题是数据记录的持续计算；这里的主问题是 GPU stage 的异构服务时间、
  KV/cache 位置、codec context、播放缓冲和软实时 SLO。
- Spark micro-batch 的固定 trigger interval 对 LLM 太粗。GPU 需要 microbatch，但 flush
  条件应该是 deadline-aware，而不是每 N ms 固定切。
- Flink event-time watermark 解决 out-of-order correctness；LLM 的 watermark 更像
  progress/deadline signal，用来决定哪些 chunk 该先算、哪些可以 merge、哪些已经 miss。
- Exactly-once sink 不是 research prototype 的第一目标。第一目标是 monotonic append、
  bounded memory、deadline trace 和可重复实验。

### 建议的核心接口

第一版 runtime 可以非常小，但接口要像 streaming engine：

```python
class StreamOperator(Protocol):
    async def open(self, ctx: OperatorContext) -> None: ...
    async def process_event(self, event: StreamEvent, out: OutputCollector) -> None: ...
    async def on_timer(self, timer: StreamTimer, out: OutputCollector) -> None: ...
    async def snapshot_state(self, barrier: CheckpointBarrier) -> OperatorState: ...
    async def close(self) -> None: ...


class StageActor(StreamOperator):
    async def build_batch(self, events: list[StreamEvent]) -> ModelBatch: ...
    async def run_batch(self, batch: ModelBatch) -> list[StreamEvent]: ...
```

对应的数据结构：

```text
StreamEvent:
  key = (session_id, stream_id)
  seq_no
  graph_id, edge_id, producer, consumer
  kind = input | prefill | decode | codec | waveform | control | watermark | barrier
  event_time = token_index | audio_time_ms | video_frame_idx
  deadline
  payload_ref
  credits_required
  cost_hint

OperatorState:
  keyed_state[session_id]
  operator_state
  pending_timers
  credit_state
```

这里 `StageActor` 是 Flink `StreamTask` 和 Spark physical operator 在 LLM 场景里的窄化版：
它有 lifecycle 和 state，但执行核心仍然调用现有 vLLM `EngineCore` / model runner。

### 从哪里开始 build

不要一开始就接多进程 transport 或 OpenAI API。先做一个小到能完全测清楚的 in-process
streaming engine：

1. **runtime skeleton**：实现 `StreamEvent`、`StreamOperator`、`OutputCollector`、
   `KeyedStateStore`、`EdgeQueue`、`CreditState`。
2. **mailbox loop**：每个 `StageActor` 一个 mailbox，循环处理 data event、timer event、
   credit return 和 barrier。这个结构借鉴 Flink task thread，避免 processor、scheduler、
   connector 到处直接改状态。
3. **microbatch builder**：每个 stage 有一个 `BatchBuilder`，输入是 ready events，flush
   条件是 earliest deadline、max batch size、max wait、GPU slot 和 downstream credit。
4. **policy layer**：实现 FIFO、EDF、EDF+credit 三个 scheduler，以及 `emit | hold | merge`
   trigger policy。
5. **Qwen3-TTS preset**：`TextSource -> TalkerActor -> CodecPacker -> Code2WavActor -> AudioSink`。
   先不进 OpenAI server，直接 research entrypoint。
6. **metrics and trace**：每个 event 记录 enqueue、schedule、run、emit、consume 时间戳；
   输出 TTFA、underrun、queue depth、credit、batch size、stage utilization。
7. **fault model 暂缓**：checkpoint barrier 第一版只用来切 trace 和 dump in-memory state；
   等 policy 实验证明有效后，再考虑恢复、savepoint 和 multi-replica。

这个路线的好处是：Flink/Spark 的 runtime 纪律留下来了，但工程量仍然像一个 research
prototype，而不是重写一个通用分布式流处理系统。

参考来源：Flink 的
[Stateful Stream Processing](https://nightlies.apache.org/flink/flink-docs-release-2.2/docs/concepts/stateful-stream-processing/)、
[Timely Stream Processing](https://nightlies.apache.org/flink/flink-docs-release-2.2/docs/concepts/time/)、
[Task Lifecycle](https://nightlies.apache.org/flink/flink-docs-release-2.2/docs/internals/task_lifecycle/)，
以及 Spark 的
[Structured Streaming Programming Guide](https://spark.apache.org/docs/3.5.7/structured-streaming-programming-guide.html)。

## 哪些复用

research prototype 应只复用会直接缩短实现路径、且不会把旧 request lifecycle 带进来的部分：

- Qwen3 模型代码、tokenizer、weight loading、sampling params 和基础 config parsing。
- vLLM `EngineCore`、worker、model runner、attention kernel、KV/block manager、CUDA graph
  capture 等 GPU 执行基础设施。
- 现有 stage model runner，作为 `StageActor` 的 backend。第一版不重写 CUDA execution，
  只在上层改变 work 组织和调度。
- stage processor 中模型特定的 tensor pack/unpack、speaker/language metadata extraction、
  Code2Wav context-window 构造。要把这些逻辑拆成 policy-free helper。
- `SharedMemoryConnector` 和 #3509 风格 transport hardening 可以稍后接入。第一版优先用
  in-process tensor refs / queues，先把 scheduler 和 trigger 结果跑出来。
- 现有 benchmark harness 中可复用的 workload、client、结果输出和 perf tracing。

暂时不要复用这些作为 prototype 的主路径：

- general stage config / pipeline registry 的完整兼容层。
- `AsyncOmniEngine` 的 public API byte compatibility。
- `OmniSchedulingCoordinator` / `OmniChunkTransferAdapter` 的 lifecycle ownership。
- 复杂 connector wire protocol 和多进程 fault-isolation 语义。

## 哪些抛弃或旁路

旧 orchestrator 不需要现在删除，但 research path 不应该继续沿着它改：

- **旁路 Orchestrator request progression**：
  `Orchestrator._route_output`、`_forward_to_next_stage`、`_prewarm_async_chunk_stages`
  不进入 prototype 主链路。新路径由 stream graph activation 决定下游 work，而不是
  `stage_id + 1` 和 placeholder prompt length。
- **丢掉 `WAITING_FOR_CHUNK` 作为主抽象**：
  以 request status 表示 chunk readiness 太粗。一个 session 可能同时在多个 edge 上有
  ready、blocked、merged、expired chunk。
- **丢掉 full-payload / async-chunk 双路径兼容压力**：
  full payload、async chunk、terminal sentinel、session update 都表示为 `StreamEvent`。
  不在 prototype 里维护两套 coordinator。
- **processor 不再决定 trigger policy**：
  `talker2code2wav_async_chunk` 这类函数拆成 `collect_state(...)`、
  `pack_window(...)` 和 `emit_event(...)`。何时发、发多大、是否 merge 由 policy 决定。
- **不保留 server-level API 互斥**：
  `/v1/realtime` 和 TTS/chat 都是 session，只是生命周期不同。prototype 可以先提供
  research entrypoint，不急着做 OpenAI API 完整兼容。
- **graph 先硬编码**：
  只写 Qwen3-TTS、Qwen3-Omni、interactive 三个 preset。不要先做通用 YAML graph
  compiler。
- **平台和模型范围收窄**：
  第一版只追 CUDA + Qwen3/TTS/interactive。PD、CFG、diffusion、NPU/XPU、多 replica
  都推迟。
- **不允许无界队列**：
  producer 没有 downstream credit 时不能继续 emit，terminal/control event 例外。

## 建议的新目录

建议把 research prototype 放在独立 package，避免半改旧 engine：

```text
vllm_omni/streaming_research/
  __init__.py
  event.py          # StreamEvent, PayloadRef, StreamId, Deadline
  graph.py          # hardcoded graph preset and edge metadata
  operator.py       # StreamOperator, OutputCollector, OperatorContext
  state.py          # in-memory keyed/operator/credit state
  runtime.py        # mailbox loop and event dispatch
  batcher.py        # deadline-aware GPU microbatch builder
  policies.py       # trigger, credit, deadline, overload policies
  scheduler.py      # FIFO/EDF/weighted EDF over StreamEvent
  stage_actor.py    # thin wrapper over existing EngineCore/model runner
  engine.py         # ResearchStreamEngine public surface
  session.py        # bounded request and unbounded session lifecycle
  metrics.py        # continuity and SLO-goodput metrics
```

外部接口先保持窄：

```python
engine = ResearchStreamEngine.from_preset("qwen3_tts")

async for event in engine.stream_tts(
    text=text,
    voice=voice,
    slo_class="interactive",
):
    yield event
```

这样可以在不碰 OpenAI server compatibility 的情况下跑实验，也能把旧 orchestrator 保留为
baseline。

## 分阶段计划

### Phase 0: Metrics And Baseline

先建立研究判断标准，而不是先写大 runtime。

- 增加 continuity metrics：TTFA、underrun p50/p95/p99、playback buffer depth、chunk
  arrival gap、emitted chunk size、per-stage queue depth、per-stage service time。
- 复现 Qwen3-TTS Base voice_clone / CustomVoice 在 c=32/64/128/256 下的当前结果。
- 复现 Qwen3-Omni audio/video 在 c=1/4/10/32 下的 TTFA、ITL、RTF 和 stage overlap。
- 增加 trace，把 transport wait、scheduler wait、stage compute、D2H/H2D、client
  playback underrun 分开。

退出标准：一个命令能生成 baseline 表，能解释为什么 underrun p99 会到秒级。

### Phase 1: Qwen3-TTS Talker -> Code2Wav Minimal Loop

第一刀只做最容易出结果的链路。

- 新增 `vllm_omni/streaming_research/` 的核心 data types 和 in-process queue。
- 建 `ResearchStreamEngine.from_preset("qwen3_tts")`。
- 写 `TalkerActor` 和 `Code2WavActor`，底层调用现有 model runner / EngineCore。
- 把 `talker2code2wav_async_chunk` 拆出 policy-free 的 collect/pack helper。
- 先实现 FIFO、EDF、EDF+simple-credit 三个 policy。

退出标准：不走旧 orchestrator，也能跑通一条 Qwen3-TTS stream，并输出音频 chunk。

### Phase 2: Credit And Deadline Policy

在最小链路上验证 research claim。

- 每条 edge 维护 `available_chunks`、`available_bytes`、`target_buffer_ms`。
- deadline 先从 playback model 生成：first chunk 用 TTFA target，steady-state 用
  expected playback time + buffer target。
- 实现 `emit | hold | merge` trigger decision。
- overload 时先做 admission control，再对非 interactive stream 降级或合并 chunk。
- 记录 policy trace：每个 chunk 为什么发、为什么 hold、为什么 merge。

退出标准：同一 GPU budget 下，EDF+credit 的 SLO goodput 高于 FIFO/round-robin，并且
queue depth 有界。

### Phase 3: Qwen3-Omni Talker -> Code2Wav

把已经验证的 policy 搬到 Qwen3-Omni 的音频输出边。

- 先保留完整 Thinker prefill barrier，只处理 Talker 到 Code2Wav 的 stream edge。
- 复用 Qwen3-Omni processor 的 tensor pack/unpack，但移出 fixed `codec_chunk_frames`
  trigger。
- 对比 Qwen3-TTS 和 Qwen3-Omni 的 service-time distribution，确认 policy 是否需要
  model-specific cost hint。

退出标准：Qwen3-Omni audio 输出能在 research engine 下用相同 scheduler 跑通。

### Phase 4: Thinker -> Talker Handoff

把跨 stage handoff 显式表示为 stream event，但先不做 partial prefill。

- Thinker 完整 prefill 完成后发 `StreamEvent(kind=prefill_ready)` 给 Talker。
- Talker actor 消费事件并启动 decode / codec stream。
- 用 session state 记录 text/audio/video input、Thinker output、Talker state 和终止条件。

退出标准：Qwen3-Omni 的 Thinker -> Talker -> Code2Wav graph 在新 engine 中端到端跑通。

### Phase 5: Interactive Session Skeleton

支持 Thinking Machines Lab interactive 这类长生命周期 workloads。

- 把 bounded TTS request 和 unbounded interactive session 都建模为 `StreamSession`。
- session input append/revise/cancel 都是 control event。
- scheduler 同时处理短请求和长 session 的 chunk，使用同一套 credit/deadline 逻辑。
- 第一版只保证单进程内正确性，不追求 OpenAI realtime API byte compatibility。

退出标准：一个 interactive session 和一批短 TTS request 可以在同一 research engine 中混跑。

### Phase 6: Cross-Stage Prefill Streaming

最后再攻最有研究价值、也最容易踩质量坑的部分。

- 把 encoder/Thinker prefill output 拆成 incremental semantic chunk。
- Talker prefill 基于 partial Thinker state 启动，并带 dependency range。
- policy 根据剩余 prompt length、预期 TTFA 收益、downstream load 和质量风险决定是否 emit。
- 对 audio/video input 做 quality/WER/alignment check，特别关注 future context 改变语义的情况。

退出标准：长 audio/video prompt 的 TTFA 改善，同时 selected quality tests 没有可测回退。

### Phase 7: Backport Or Production Bridge

只有在 prototype 结果足够好之后再考虑接回生产路径。

- 把 `StageActor` 接到已有 multi-process transport。
- 接 #3509 的 async D2H/H2D、pinned pool、tensor blob。
- 做 OpenAI API compatibility bridge。
- 做 multi-replica sticky routing、fault isolation、4h stability。

## 4 周 Prototype Milestone

| 周 | 目标 | 关键产物 |
| --- | --- | --- |
| Week 1 | Qwen3-TTS Talker -> Code2Wav 最小闭环 | `streaming_research` skeleton，FIFO/EDF ablation，能输出音频 chunk |
| Week 2 | credit/deadline/overload | 有界 queue，policy trace，SLO goodput 表 |
| Week 3 | Qwen3-Omni audio 输出 | Talker -> Code2Wav adapter，共用 scheduler，TTS/Omni 对比实验 |
| Week 4 | interactive session skeleton | `StreamSession`，control event，短 TTS + 长 session 混跑 |

## 原计划中被推迟的内容

这些内容重要，但不应该挡住 research prototype：

- generic YAML graph compiler 和 arbitrary pipeline compatibility。
- 完整 OpenAI API compatibility。
- multi-replica serving、sticky routing、跨 replica credit sharing。
- PD / CFG / diffusion pipeline。
- NPU/XPU/多平台支持。
- tensor_blob/pinned SHM/低拷贝 wire path 的完整生产实现。
- 4h stability、生产级 fault isolation、legacy rollback。

## Motivation 实验

这些实验应在实现前和实现过程中持续运行，而不是最后才补。

1. **Continuity baseline**：Qwen3-TTS Base voice_clone 和 CustomVoice，在
   c=32/64/128/256 下测 underrun p95/p99、TTFA、RTF、request goodput、stage
   utilization、queue depth。
2. **Trigger ablation**：固定 `initial_codec_chunk_frames` 为 `{1,2,4,8,16,25}`、
   当前 dynamic IC、deadline-aware IC、credit-aware merge。指标：TTFA、underrun、
   Code2Wav compute amplification、audio quality。
3. **Scheduler ablation**：旧路径 request order / round-robin、按 chunk arrival FIFO、
   EDF、weighted EDF、EDF+credit、EDF+admission control。需要混合 SLO class。
4. **Credit ablation**：unbounded queue、static credit、playback-buffer adaptive credit、
   consumer-service-time adaptive credit。指标：queue depth、memory、underrun、
   producer idle time。
5. **Transport versus policy isolation**：baseline、仅 #3509 风格 async D2H/H2D、仅
   policy、combined。用于证明主要 gap 来自 scheduling / trigger，而不是 copy overhead。
6. **Qwen3-Omni transfer**：在保留完整 Thinker barrier 的前提下，只替换 Talker ->
   Code2Wav policy，测 TTFA、ITL、RTF、underrun、quality。
7. **Cross-stage prefill barrier**：完整 Thinker prefill 后 Talker、固定 chunked prefill、
   deadline-triggered prefill。用长 audio/video input 测 TTFA、ITL、quality/WER 和
   stage overlap。
8. **Unified sessions**：短 TTS request、长 chat+audio request 和 interactive session
   混跑。验证没有 API-mode exclusion，并测 fairness。
9. **Overload behavior**：打到超过系统 capacity，验证 admission/downgrade/merge 能提升
   SLO goodput，而不是让所有 stream 一起 miss。

主结果指标应是 **SLO goodput**：每秒满足所有相关约束的 request 或 active session 数。
约束包括 TTFA、steady-state underrun、text ITL、RTF 和 success rate。

## 开放问题

- `StageActor` 能多直接地包住现有 `EngineCore` / model runner？最小可行接口是什么？
- 第一篇 paper signal 应该先押 Qwen3-TTS underrun，还是 Qwen3-Omni interactive
  end-to-end？
- 在 #3509 完整落地前，in-process tensor refs / queues 是否足够支撑 motivation 实验？
- audio playback deadline 应来自绝对播放时间、inter-chunk gap、buffer debt，还是混合模型？
- Qwen3-Omni partial prefill 对质量有多安全，尤其是 audio/video future context 会改变语义的场景？
- 哪些旧 API 行为必须在 prototype 阶段保留，哪些可以留给 backport？

## 建议的第一批 PR

1. 增加 `vllm_omni/streaming_research/` data types、simulator 和 policy unit tests。
2. 增加 Qwen3-TTS Talker -> Code2Wav research path，不走旧 orchestrator。
3. 拆分 Qwen3-TTS processor：保留 collect/pack，移出 trigger decision。
4. 增加 EDF+credit scheduler、policy trace 和 continuity metrics。
5. 增加 Qwen3-Omni Talker -> Code2Wav adapter，复用同一 scheduler。
