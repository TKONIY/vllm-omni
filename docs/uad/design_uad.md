# UAD 统一引擎设计

目标：基于 vLLM v1 AR EngineCore 构建 unified AR + DiT engine，使同一个 DP group 能在同一套 v1 request lifecycle、scheduler/executor/worker 边界内处理 AR token 和 DiT step work。第一目标模型是 HunyuanImage3。

当前代码不是早期 nano-UAD toy engine。当前设计以 vLLM v1 core 兼容为硬约束：能占用 v1 slot 的 UAD 类继承 v1 interface，v1 方法保持原签名和原语义，UAD 扩展只以 additive subclass fields 或额外方法承载。

## 1. 当前接入点

```text
vllm-omni serve --uad-engine
  -> VLLM_OMNI_USE_UAD_ENGINE=1
  -> StageEngineCoreProc.run_stage_core()
  -> resolve_stage_engine_core_cls(...)
  -> UADEngineCore(StageEngineCoreProc)
```

`UADEngineCore` 保留 vLLM v1 EngineCoreProc 进程边界和 wire protocol：

- 输入仍是 v1 `EngineCoreRequest`。
- 输出仍是 v1 `EngineCoreOutputs`。
- `self.scheduler` 仍指向原生 v1 scheduler 实例。
- `self.model_executor` 仍指向原生 v1 executor 实例。
- UAD 只在 `UADEngineCore.step()` 内用 `self.uad_scheduler`、`self.uad_executor`、`self.uad_runner` 串起扩展路径。

这样做的原因是 v1 core 的 add/abort/pause/reset/stats/KV connector/health 等辅助路径会直接访问 `self.scheduler` 和 `self.model_executor`，这些路径必须继续看到原生 v1 interface 语义。

## 2. 继承关系

当前 UAD 类按 v1 interface 收紧：

| UAD 类 | v1 父类 | 语义 |
|---|---|---|
| `UADEngineCore` | `StageEngineCoreProc` | 覆盖 step orchestration，保留 EngineCoreProc 协议 |
| `UADScheduler` | `SchedulerInterface` | 组合 `base_scheduler`，只代理 UAD step 需要的 v1 scheduler 方法 |
| `UADSchedulerOutput` | `SchedulerOutput` | 保留全部 v1 scheduler output 字段，额外加 `uad_items` |
| `UADExecutor` | `Executor` | 组合 `base_executor`，只代理 UAD step 需要的 v1 executor 方法 |
| `UADModelRunnerOutput` | `ModelRunnerOutput` | 保留全部 v1 model runner output 字段，额外加 `phase_outputs` |
| `UADGPUWorker` | `WorkerBase` | 未来 UAD worker backend 的 v1-compatible placeholder |

不继承 v1 的对象只作为 UAD 内部结构，例如 `UADRequestState`、`UADScheduleItem`、`UADPhaseOutput`。

## 3. Step Flow

当前 `UADEngineCore.step()`：

```text
UADEngineCore.step()
  -> UADScheduler.schedule()                  # returns UADSchedulerOutput(SchedulerOutput)
  -> UADExecutor.execute_model(...)           # accepts SchedulerOutput-compatible object
  -> UADScheduler.get_grammar_bitmask(...)
  -> future.result()
  -> UADExecutor.sample_tokens(...) if needed
  -> UADRunner.process_outputs(...)           # returns UADModelRunnerOutput(ModelRunnerOutput)
  -> UADScheduler.update_from_output(...)
  -> EngineCoreOutputs
```

当前 AR path 是 passthrough：`UADScheduler` 调 base scheduler，`UADExecutor` 调 base executor，`UADModelRunnerOutput` 从 base `ModelRunnerOutput` 复制字段。DiT/artifact work 还没有真实调度和执行，只保留接口位置。

## 4. Scheduler Contract

`UADScheduler` 是 v1 `SchedulerInterface` 的实现，但不是完整 v1 scheduler replacement。当前采用 subclass + composition，只实现 `UADEngineCore.step()` 需要的最小路径：

```python
class UADScheduler(SchedulerInterface):
    def __init__(self, base_scheduler: SchedulerInterface) -> None:
        self.base_scheduler = base_scheduler

    def schedule(self) -> UADSchedulerOutput:
        base_output = self.base_scheduler.schedule()
        uad_items = ...  # TODO
        return UADSchedulerOutput.from_base(base_output, uad_items)
```

UAD step path 的 v1 methods 必须保留 v1 语义：

- `schedule()` 返回 `SchedulerOutput` 兼容对象。
- `get_grammar_bitmask()` 接收 `SchedulerOutput`。
- `update_from_output()` 接收 `SchedulerOutput` 和 `ModelRunnerOutput`。

其他 v1 lifecycle 方法，例如 `add_request()`、`finish_requests()`、`pause_state`、`reset_prefix_cache()`、`make_stats()`，在 `UADScheduler` 上明确是 unsupported stub。EngineCoreProc 的这些路径继续使用原生 `self.scheduler`，不是 `self.uad_scheduler`。

`UADSchedulerOutput` 继承 `SchedulerOutput`，额外字段：

```python
@dataclass
class UADSchedulerOutput(SchedulerOutput):
    uad_items: list[UADScheduleItem]
```

继承字段的语义不能改变。`uad_items` 只承载轻量 metadata，不直接携带大 tensor 或 CUDA tensor。

## 5. UAD Work Item

当前 `UADScheduleItem` 是 UAD 的最小调度扩展：

```python
@dataclass(frozen=True)
class UADScheduleItem:
    request_id: str
    phase: UADPhase
    num_scheduled_tokens: int
    input_kind: Literal["ar_tokens", "dit_latent_step", "artifact_decode"]
    output_kind: Literal["sample_tokens", "denoise_pred", "artifact", "none"]
    needs_kv_slots: bool = False
    num_persistent_tokens: int = 0
    step_index: int | None = None
    total_steps: int | None = None
    shape_bucket: tuple[int, ...] | None = None
```

语义：

- AR work 继续由 v1 `SchedulerOutput.num_scheduled_tokens` 表示。
- DiT/artifact work 用 `uad_items` 表示。
- `num_scheduled_tokens` 对 UAD item 表示该 item 对调度 budget 的 token-like cost。
- `num_persistent_tokens` 表示本 item 中会 commit 成后续 reusable engine context/KV 的 token-like 数量。非 final DiT denoise 为 0；final image context commit 为 image context token 数。
- `needs_kv_slots` 表示该 work 是否需要物理 KV slot/slot mapping。它通常由 `num_persistent_tokens > 0` 推导，但保留为 executor metadata，避免把逻辑进度和物理执行细节混成一个变量。
- `step_index`、`total_steps`、`shape_bucket` 用于 DiT step batching。

`num_scheduled_tokens` 和 `num_persistent_tokens` 必须分开：前者是本 tick 调度/算力 budget，后者是执行成功后可提交到后续 context 的 token 数。对 DiT 来说，很多 step 会消耗完整 `num_scheduled_tokens`，但 `num_persistent_tokens=0`。

## 6. Request State

`UADRequestState` 是 UAD 附加在 vLLM request lifecycle 上的状态，不替代 v1 `Request`：

```python
class UADPhase(str, Enum):
    AR_PREFILL = "ar_prefill"
    AR_DECODE = "ar_decode"
    DIT_STEP = "dit_step"
    ARTIFACT_DECODE = "artifact_decode"

@dataclass
class UADRequestState:
    request_id: str
    phase: UADPhase
    engine_tokens: list[UADToken]
    materialized_tokens: list[UADToken]
    dit_step_index: int
    dit_num_steps: int
    dit_query_tokens: int
    runtime_state: dict[str, Any]
```

字段语义：

- `engine_tokens`：后续模型可见的统一多模态 token ledger。
- `materialized_tokens`：对外 streaming / detokenize / artifact 输出 ledger。
- `phase`：AR、DiT、artifact decode 的请求状态。
- `dit_*`：DiT step 执行和 image token 数相关 metadata。
- `runtime_state`：模型私有状态，例如 HunyuanImage3 的 size、seed、CFG、latent shape。

当前代码还没有把 `UADRequestState` 正式挂到 v1 `Request` side table；这是下一步 scheduler state transition 的工作。

### 6.1 状态所有权

UAD 不复制 v1 已经维护的 AR 状态。状态 ownership 按下面划分：

| 状态 | Owner | 更新时机 |
|---|---|---|
| `Request.prompt_token_ids` | v1 request | request 创建时 |
| v1 generated token ledger | v1 scheduler/request | base scheduler 消费 `ModelRunnerOutput.sampled_token_ids` 时 |
| `Request.num_computed_tokens` | v1 scheduler/KV path | reusable KV/context commit 后 |
| `SchedulerOutput.num_scheduled_tokens` | v1 scheduler | 每次 AR prefill/decode schedule |
| `ModelRunnerOutput.sampled_token_ids` | v1 model runner/sampler | AR sampling 后 |
| `UADScheduleItem.num_persistent_tokens` | UAD scheduler | schedule DiT/artifact item 时给出预期 persistent token 数 |
| `UADRequestState.phase` | UAD scheduler/state machine | phase transition 后 |
| `UADRequestState.engine_tokens` | UAD state | 新多模态 context token materialize 为 engine-visible token 时 |
| `UADRequestState.materialized_tokens` | UAD state/output path | 对外 text/image/audio/artifact 输出产生时 |
| `UADRequestState.dit_*` | UAD state | 进入 DiT phase 和每个 DiT step 后 |
| `UADRequestState.runtime_state` | 模型私有 UAD state | model-specific metadata 变化时 |

因此，当前 UAD state 不再单独维护旧文档里的 `ar_sampler_token_ids`。AR sampler history 继续由 v1 request/scheduler/model runner 的原生字段维护；UAD 只在需要多模态 phase 切换或 artifact 输出时记录额外状态。

### 6.2 UADToken

```python
@dataclass(frozen=True)
class UADToken:
    modality: Literal["text", "image", "video", "audio", "latent", "control"]
    token_id: int | None = None
    payload: Any | None = None
```

字段语义：

- `modality`：token 的逻辑模态，用于区分 text token、image context token、latent/control token 等。
- `token_id`：能映射到 tokenizer/model vocab 的 token id。纯 artifact 或 latent payload 可以为 `None`。
- `payload`：非 token-id 数据，例如 image artifact handle、latent metadata、shape/seed/control 信息。不能直接放大 tensor；大对象应走外部 artifact/cache handle。

### 6.3 UADRequestState 字段

| 字段 | 语义 | 何时更新 |
|---|---|---|
| `request_id` | v1 request id | 创建 UAD state 时固定 |
| `phase` | 当前 UAD phase | AR 输出触发边界 token、DiT step 完成、artifact decode 完成时 |
| `engine_tokens` | 后续模型可见的统一多模态 logical tokens | AR 产生 engine-visible control token；final DiT commit image context token；未来 multiturn 输入追加 |
| `materialized_tokens` | 对外可见输出 ledger | text streaming token、image artifact、audio/video artifact 产生时 |
| `dit_step_index` | 已完成或当前准备执行的 DiT step index | 每个 DiT step 完成后推进 |
| `dit_num_steps` | 当前 DiT job 总 step 数 | 进入 DiT phase 时由模型配置/请求参数确定 |
| `dit_query_tokens` | 当前图像 latent/image context 的 token-like 数量 | 进入 DiT phase 时由 resolution/patch/latent shape 确定 |
| `runtime_state` | 模型私有状态 | 进入 DiT phase、CFG 参数变化、shape bucket/seed/guidance 等变化时 |

`engine_tokens` 和 v1 `Request.num_computed_tokens` 不是同一个变量。`engine_tokens` 是 UAD 的 logical ledger；`Request.num_computed_tokens` 是 v1 对 reusable computed prefix/KV 的进度。只有当 UAD token 真的 commit 到后续可复用 engine state 时，才应推进 v1 computed-token 语义。

### 6.4 Phase Update

runner 不直接改 request state。runner 输出 UAD phase delta：

```python
@dataclass
class UADPhaseUpdate:
    new_engine_tokens: list[UADToken]
    new_materialized_tokens: list[UADToken]
    num_new_computed_tokens: int
    next_phase: UADPhase | None
    runtime_state_delta: dict[str, Any]

@dataclass
class UADPhaseOutput:
    request_id: str
    phase: UADPhase
    update: UADPhaseUpdate
    raw_output: Any | None = None
```

字段语义：

- `new_engine_tokens`：追加到 UAD logical engine context 的 tokens。
- `new_materialized_tokens`：追加到对外输出 ledger 的 tokens/artifacts。
- `num_new_computed_tokens`：本次 UAD update 中需要推进 reusable computed-prefix 语义的 token 数。非 final DiT denoise 通常为 0；final image context commit 应为 image context token 数。
- `next_phase`：状态机决定的下一 phase；为 `None` 表示保持当前 phase。
- `runtime_state_delta`：模型私有状态增量，例如 latent shape、timestep schedule、CFG metadata。
- `raw_output`：debug/profiling 用的原始输出引用；生产路径不应依赖它传大 tensor。

### 6.5 典型状态变化

| 事件 | v1 状态变化 | UAD 状态变化 |
|---|---|---|
| request add | v1 创建 `Request`，prompt/mm features 进入原生 request | 创建 `UADRequestState(phase=AR_PREFILL)` |
| AR prefill/decode schedule | base scheduler 更新 scheduled tokens/KV allocation | 通常不变 |
| AR sample 普通 text token | base scheduler append generated token，更新 v1 output | 可按 output policy 追加 `materialized_tokens` |
| AR sample `<img_ratio_*>` | base scheduler 仍按普通 generated token 处理 | state machine 设置 `phase=DIT_STEP`，填 `dit_num_steps/dit_query_tokens/runtime_state` |
| DiT non-final step | 不推进 v1 sampled token；不 commit v1 KV | `dit_step_index += 1`，通常 `num_new_computed_tokens=0` |
| DiT final step | 需要后续实现 image context KV/cache commit | 追加 image `engine_tokens`，推进 phase 到 artifact 或 AR |
| artifact/VAE decode | 不进入 v1 token progress | 追加 image artifact 到 `materialized_tokens` |
| multiturn next AR | v1 应看到已 commit 的 text+image context | `engine_tokens` 保留完整多模态 context ledger |

## 7. Executor / Runner Contract

`UADExecutor(Executor)` 当前组合 `base_executor`：

- `execute_model(SchedulerOutput, non_block)` 直接 delegate 到 base executor。
- `sample_tokens(GrammarOutput, non_block)` 直接 delegate 到 base executor。
- `collective_rpc()`、`check_health()` 等 v1 executor lifecycle 方法在 `UADExecutor` 上明确是 unsupported stub。
- `max_concurrent_batches` 暂时固定为 1，因为 `UADEngineCore` 当前禁用 upstream batch queue。

后续 DiT execution 会在这个边界扩展，但不能改变 v1 executor 方法签名。

`UADRunner` 当前不是 v1 `GPUModelRunner`，而是 EngineCore 内部 output processor boundary：

```text
raw ModelRunnerOutput
  -> UADRunner.process_outputs(...)
  -> UADModelRunnerOutput(ModelRunnerOutput)
```

`UADModelRunnerOutput` 继承 `ModelRunnerOutput`，额外字段：

```python
@dataclass
class UADModelRunnerOutput(ModelRunnerOutput):
    phase_outputs: list[UADPhaseOutput]
```

base scheduler 可以继续按普通 `ModelRunnerOutput` 消费它。UAD scheduler 后续可以读取 `phase_outputs` 应用 phase transition。

## 8. KV / Token Progress

当前设计继续复用 v1 scheduler 和 KV/cache/block table 体系，不自研 production KV manager。

原则：

- AR prefill/decode 的 logical progress 仍由 v1 `Request.num_computed_tokens`、`SchedulerOutput.num_scheduled_tokens`、KV block allocation 维护。
- UAD DiT item 如果最终要进入 multiturn context，必须在某个 commit step 以 v1-compatible 方式体现为新增 engine tokens 和 KV/cache state。
- 非最终 DiT denoise step 可以消耗 compute budget，但不应伪装成已经提交到 reusable KV 的 token。
- artifact/VAE decode 不进入 v1 token progress；它属于 materialized output path。

UAD token persistence 的两阶段语义：

| 字段 | 所在对象 | 语义 |
|---|---|---|
| `num_persistent_tokens` | `UADScheduleItem` | scheduler 对本 item 可提交 token 数的预期，用于 KV/slot allocation 和执行 metadata |
| `num_new_computed_tokens` | `UADPhaseUpdate` | runner/state update 实际确认要推进 reusable computed-prefix 语义的 token 数 |

第一版可以要求两者相等；后续如果出现 preemption、partial commit 或 execution failure，`num_persistent_tokens` 是 schedule-time intent，`num_new_computed_tokens` 是 update-time fact。

当前代码还没有实现 DiT commit、KV slot allocation、attention metadata 或 multiturn image context commit。

## 9. HunyuanImage3 MVP 方向

HunyuanImage3 第一版应按 v1-compatible path 做最小闭环：

1. AR 仍走 v1 scheduler/executor/runner。
2. 识别 `<img_ratio_*>` 后，给 request 附加 UAD phase state。
3. scheduler 在后续 tick 生成 `UADScheduleItem(phase=DIT_STEP)`。
4. executor/worker 执行 DiT step，输出 `UADPhaseOutput`。
5. scheduler 应用 `UADPhaseOutput`，推进 DiT step。
6. final DiT step 默认 commit image context tokens/KV，以支持 multiturn。
7. VAE/artifact decode 走 materialized output，不污染 v1 AR sampler token history。

## 10. 当前未实现

- UAD request state side table。
- `<img_ratio_*>` 到 DiT phase 的真实 transition。
- DiT scheduler budget、shape bucket 和 step batching。
- DiT executor/worker/model runner。
- full dense / mixed attention metadata。
- final image context KV commit。
- artifact output processor。
- CFG parallel。
- SP / sequence parallel。
- AR + DiT FFN/MoE hidden 合批。

## 11. 设计约束

- 不把非 `SchedulerOutput` wrapper 塞进 v1 scheduler/executor path。
- 不改变 v1 method signature。
- 不改变 inherited v1 fields 的语义。
- UAD metadata 必须 additive。
- 需要替换 v1 slot 的类应继承对应 v1 interface。
- UAD 内部状态类不需要继承 v1。
