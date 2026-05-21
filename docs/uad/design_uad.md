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

## 9. State Machine

UAD request state machine 是 scheduler-owned state。phase 只表示下一次可调度 work 的类型，不表示每个 Python 后处理函数都要变成 scheduler-visible phase。

| 当前 phase | 触发 | 下一 phase | 关键更新 |
|---|---|---|---|
| `AR_PREFILL` | prompt prefill 完成 | `AR_DECODE` | v1 computed-token/KV 进度更新 |
| `AR_DECODE` | 普通 text token | `AR_DECODE` | v1 sampled token ledger 更新；可产生 materialized text |
| `AR_DECODE` | `<img_ratio_*>` | `DIT_STEP` | 写入 image size、`dit_query_tokens`、`dit_num_steps`、latent/noise seed |
| `DIT_STEP` | non-final denoise step 完成 | `DIT_STEP` | `dit_step_index += 1`；不推进 computed-token/KV 语义 |
| `DIT_STEP` | final denoise step 完成 | `ARTIFACT_DECODE` | commit image context tokens/KV；追加 `engine_tokens` |
| `ARTIFACT_DECODE` | artifact 生成完成 | finished 或下一轮 AR | 追加 `materialized_tokens` |

`ARTIFACT_DECODE` 类似输出 materialization，不等价于 v1 text sampling。它是否由 scheduler 感知取决于是否占用 GPU/model budget；纯 CPU/IO 后处理可以从 scheduler 隐藏。

## 10. Scheduler 语义

UAD scheduler 仍按 token-like budget 工作，不引入独立的 phase budget。

| Work | 表示方式 | 调度粒度 |
|---|---|---|
| AR prefill/decode | inherited `SchedulerOutput` 字段 | 复用 v1 chunked prefill / decode 语义 |
| DiT step | `UADScheduleItem` | 一个 denoise step 是一个调度单元 |
| artifact decode | `UADScheduleItem` 或 scheduler 外后处理 | 取决于是否占 GPU/model budget |

DiT step 的 `num_scheduled_tokens` 通常等于该 step 的 image query token 数。第一版不把一个 DiT denoise step 再切成任意小 chunk，因为 timestep、latent、CFG 和 image query 是耦合的；除非具体模型显式支持 partial step。

同一个 `UADSchedulerOutput` 可以同时包含 AR work 和 DiT/artifact `uad_items`。executor 可以按 attention 约束拆 kernel，但 FFN/MoE 只要 hidden size、dtype、layer、TP/EP mesh 和 backend 兼容，就应尽量合批。

## 11. Attention 实现

第一版只保留一种方案：复用 vLLM causal paged attention，再补 DiT chunk 内 bidirectional attention。

有效 attention 语义：

- 所有 token 都能按 causal 规则看见自己的前文，这部分与 vLLM paged attention 一致。
- AR token 没有 bidirectional patch，causal output 就是最终 attention output。
- DiT 当前 step 的 image query tokens 除了看前文，还需要在本 request 的当前 DiT chunk 内做 bidirectional attention。
- 不同 request 的 DiT chunks 之间互不可见。

执行形态：

```text
1. 用 vLLM attention metadata / block table / slot mapping 跑 causal paged attention
2. 对每个 DiT chunk 跑 chunk-internal bidirectional attention
3. 在 softmax score / logsumexp 语义下合并 causal 部分和 bidirectional 部分
```

第三步不能简单把两个 attention output 相加；需要保留 softmax normalization。实现上可以先用 reference mixed attention 验证语义，再替换成高性能 kernel。

不单独引入 `attention_signature` 字段。第一版由 `phase`、`input_kind`、DiT chunk range、slot mapping 和 shape metadata 推导 attention 行为。

## 12. Executor / Runner 实现约束

`UADInputs` 应由 runner 组装，内容取决于 work 类型：

| Work | 输入 |
|---|---|
| AR | token ids / positions / block tables / slot mapping |
| DiT | latent or image-query embeds、timestep、positions、context KV/block tables、DiT chunk ranges |
| artifact | VAE/artifact decode 所需 handle 或 latent |

executor 可以拆 attention，但不应该无理由拆 FFN/MoE。目标是让 AR hidden 和 DiT hidden 在兼容层上形成更大的 FFN/MoE batch。

output 处理规则：

- AR sample 输出进入 v1 sampled-token 路径。
- DiT denoise 输出进入 `UADPhaseOutput.raw_output` 和 `runtime_state_delta`。
- final DiT context commit 通过 `new_engine_tokens` 和 `num_new_computed_tokens` 推进 UAD/v1 reusable context。
- artifact decode 输出进入 `new_materialized_tokens`，用于 streaming 或最终 response。

## 13. KV / Persist Commit

`num_scheduled_tokens` 是 compute budget；`num_persistent_tokens` 是 schedule-time commit 预期；`num_new_computed_tokens` 是 update-time commit 事实。

DiT non-final step 不产生 reusable context/KV，因此：

```text
num_scheduled_tokens = image query tokens
num_persistent_tokens = 0
num_new_computed_tokens = 0
```

DiT final step 默认要 commit image context tokens/KV，以原生支持 multiturn。即使当前 response 后马上结束，也不把“不 commit”作为默认语义；是否为了内存优化跳过 commit 应是显式策略。

## 14. HunyuanImage3 Flow

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

## 15. TODO

- 连接 `UADRequestState` 和真实 v1 request lifecycle。
- 实现 `<img_ratio_*>` 到 DiT phase 的 transition。
- 实现 AR/DiT scheduler budget、shape bucket、step batching。
- 实现 DiT executor / worker / model runner。
- 实现 causal paged + chunk bidirectional attention metadata/kernel。
- 实现 final image context KV commit。
- 实现 artifact output processor。
- 讨论并实现 CFG parallel、SP / sequence parallel、AR + DiT FFN/MoE hidden 合批。
