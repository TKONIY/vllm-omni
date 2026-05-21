# UAD 统一引擎设计

目标：在 vLLM v1 EngineCore 进程边界内构建 unified AR + DiT engine，让同一个 DP group 可以调度 AR token、DiT step 和 artifact decode。第一目标模型是 HunyuanImage3。

当前代码是 scaffold，不是完整 serving path。

## 1. 接入

```text
vllm-omni serve --uad-engine
  -> VLLM_OMNI_USE_UAD_ENGINE=1
  -> StageEngineCoreProc.run_stage_core()
  -> UADEngineCore(StageEngineCoreProc)
```

`UADEngineCore` 保留 v1 EngineCore protocol：输入仍是 `EngineCoreRequest`，输出仍是 `EngineCoreOutputs`。

当前对象边界：

| 对象 | 角色 |
|---|---|
| `self.scheduler` | 原生 v1 scheduler，保留给 EngineCoreProc lifecycle 路径 |
| `self.model_executor` | 原生 v1 executor，保留给 EngineCoreProc lifecycle 路径 |
| `self.uad_scheduler` | UAD step path 的 scheduler scaffold |
| `self.uad_executor` | UAD step path 的 executor scaffold |
| `self.uad_runner` | UAD output processor boundary |

## 2. 类关系

UAD 类继承 v1 interface/base class，只用于接口可读性和 abstract method 检查；UAD 不再持有 `base_scheduler` 或 `base_executor`。

| 类 | 父类 | 说明 |
|---|---|---|
| `UADEngineCore` | `StageEngineCoreProc` | 覆盖 `step()` orchestration |
| `UADScheduler` | `SchedulerInterface` | UAD-native scheduler scaffold |
| `UADSchedulerOutput` | `SchedulerOutput` | v1 output + `uad_items` |
| `UADExecutor` | `Executor` | UAD-native executor scaffold |
| `UADModelRunnerOutput` | `ModelRunnerOutput` | v1 output + `phase_outputs` |
| `UADGPUWorker` | `WorkerBase` | future worker placeholder |

UAD 内部状态类不继承 v1，例如 `UADRequestState`、`UADScheduleItem`、`UADPhaseOutput`。

## 3. Step Flow

```text
UADEngineCore.step()
  -> UADScheduler.has_requests()
  -> UADScheduler.schedule()
  -> UADExecutor.execute_model()
  -> UADScheduler.get_grammar_bitmask()
  -> UADExecutor.sample_tokens() if AR sampling is needed
  -> UADRunner.process_outputs()
  -> UADScheduler.update_from_output()
```

当前 scaffold 行为：

- `UADScheduler.has_requests()` 返回 `False`。
- `UADScheduler.schedule()` 返回 empty `UADSchedulerOutput`。
- `UADExecutor.execute_model()` 返回 no-op future / `None`。
- `UADExecutor.sample_tokens()` 返回 empty `UADModelRunnerOutput`。

## 4. Scheduler Output

```python
@dataclass
class UADSchedulerOutput(SchedulerOutput):
    uad_items: list[UADScheduleItem]
```

约束：

- inherited `SchedulerOutput` 字段保持 v1 原语义。
- `uad_items` 只放轻量 metadata，不放大 tensor / CUDA tensor。
- AR work 继续使用 v1 `SchedulerOutput` 字段；DiT/artifact work 使用 `uad_items`。

## 5. Work Item

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

字段时机：

| 字段 | 写入时机 | 含义 |
|---|---|---|
| `num_scheduled_tokens` | schedule | 本 tick 的 token-like compute budget |
| `num_persistent_tokens` | schedule | 本 item 成功后预计 commit 到 reusable engine context/KV 的 token 数 |
| `needs_kv_slots` | schedule | executor 是否需要物理 KV slot / slot mapping |
| `step_index` / `total_steps` | schedule | DiT step batching metadata |
| `shape_bucket` | schedule | DiT shape batching metadata |

DiT non-final step 通常 `num_scheduled_tokens > 0`、`num_persistent_tokens = 0`。DiT final image-context commit 通常 `num_persistent_tokens = image context token count`。

## 6. Request State

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

字段含义：

| 字段 | 含义 |
|---|---|
| `phase` | 请求当前阶段 |
| `engine_tokens` | 后续模型可见的统一多模态 logical token ledger |
| `materialized_tokens` | 对外 streaming / artifact output ledger |
| `dit_step_index` / `dit_num_steps` | DiT 进度 |
| `dit_query_tokens` | 本图像 context 的 token-like 数量 |
| `runtime_state` | 模型私有状态，例如 image size、seed、CFG、latent shape |

UAD 不维护独立 `ar_sampler_token_ids`。AR sampler history 继续由 v1 request/scheduler/model runner 原生字段维护。

## 7. Phase Output

runner 不直接修改 request state，只返回 delta：

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

字段时机：

| 字段 | 写入时机 | 含义 |
|---|---|---|
| `new_engine_tokens` | runner output | 新增到 UAD logical engine ledger 的 token |
| `new_materialized_tokens` | runner output | 新增到对外输出 ledger 的 token/artifact |
| `num_new_computed_tokens` | runner output | 实际推进 reusable computed-prefix/KV 语义的 token 数 |
| `next_phase` | runner output | phase transition 目标 |
| `runtime_state_delta` | runner output | 模型私有状态增量 |

第一版可要求 `num_new_computed_tokens == UADScheduleItem.num_persistent_tokens`。后续支持 partial commit / preemption 时二者可以不同。

## 8. 状态所有权

| 状态 | Owner |
|---|---|
| `Request.prompt_token_ids` | v1 request |
| v1 generated token ledger | v1 scheduler/request |
| `Request.num_computed_tokens` | v1 scheduler/KV path |
| `SchedulerOutput.num_scheduled_tokens` | v1 scheduler |
| `ModelRunnerOutput.sampled_token_ids` | v1 model runner/sampler |
| `UADRequestState.phase` | UAD scheduler/state machine |
| `UADRequestState.engine_tokens` | UAD state |
| `UADRequestState.materialized_tokens` | UAD state/output path |
| `UADScheduleItem.num_persistent_tokens` | UAD scheduler |
| `UADPhaseUpdate.num_new_computed_tokens` | UAD update |

`num_computed_tokens` 只在 token 真正 commit 到后续可复用 engine state/KV 后推进。一次 forward 不必然增加 computed tokens。

## 9. HunyuanImage3 Flow

```text
request add
  -> phase = AR_PREFILL
AR emits normal text token
  -> v1 sampled token ledger updates
  -> optional materialized text output
AR emits <img_ratio_*>
  -> phase = DIT_STEP
  -> fill dit_num_steps / dit_query_tokens / runtime_state
DiT non-final step
  -> denoise state update
  -> num_new_computed_tokens = 0
DiT final step
  -> commit image context tokens/KV
  -> append image engine_tokens
artifact decode
  -> append materialized image artifact
multiturn next AR
  -> sees committed text + image context
```

## 10. TODO

- 连接 `UADRequestState` 和真实 v1 request lifecycle。
- 实现 `<img_ratio_*>` 到 DiT phase 的 transition。
- 实现 AR/DiT scheduler budget、shape bucket、step batching。
- 实现 DiT executor / worker / model runner。
- 实现 causal paged + chunk bidirectional attention metadata。
- 实现 final image context KV commit。
- 实现 artifact output processor。
- 讨论并实现 CFG parallel、SP / sequence parallel、AR + DiT FFN/MoE hidden 合批。
