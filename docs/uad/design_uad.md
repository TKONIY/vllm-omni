# UAD 统一引擎设计

目标：在 vLLM AR engine 上扩展 unified AR + DiT engine，使一个 scheduler tick 可以同时
batch AR 和 DiT 请求。第一目标模型是 HunyuanImage3。MVP 只讨论 DP + TP；CFG
parallel 和 SP 留 TODO。

核心原则：

- 复用 AR vLLM scheduler、paged KV、chunked prefill、prefix cache、continuous
  batching 和 request lifecycle。
- scheduler 继续按 token budget 工作，主账本仍是
  `SchedulerOutput.num_scheduled_tokens`。
- multimodal context 是一条持续增长的 engine token 序列；text/image/audio 等不各自
  维护位置。
- DiT denoise step 是按 token 计费的 scheduled item；VAE / artifact decode 是
  runner epilogue，不进入 scheduler token budget。

## 1. Phase

```python
class UADPhase(str, Enum):
    AR_PREFILL = "ar_prefill"
    AR_DECODE = "ar_decode"
    DIT_STEP = "dit_step"
    FINISHED = "finished"
```

| Phase | scheduler work | `num_scheduled_tokens` |
|---|---|---:|
| `AR_PREFILL` | 计算已有 prompt/context chunk | `1..chunk_limit` |
| `AR_DECODE` | 计算 1 个 query token，产出 logits | `1` |
| `DIT_STEP` | 计算 1 个 DiT denoise timestep | `dit_query_tokens` |
| `FINISHED` | 请求结束 | 不调度 |

控制面迁移不建 phase。HunyuanImage3 在 `<img_ratio_*>` 后执行
`on_phase_transition()`：append image engine tokens，初始化 latents/timesteps，设置
`phase=DIT_STEP`。这个 hook 不消耗 token budget。

## 2. Token 账本

UAD 只有两条 append-only list。

```python
@dataclass
class UADToken:
    kind: Literal["text", "image", "audio", "latent", "control"]
    token_id: int | None = None
    payload: object | None = None
    artifact: object | None = None
```

| 账本 | 用途 | 不变量 |
|---|---|---|
| `engine_tokens` | 模型后续可见的 unified context；同时承担 request 追加和 runner 下一步 token ledger | `token_id is not None`，`artifact is None` |
| `materialized_tokens` | 用户可见输出；用于 streaming / detokenize / artifact output | `payload is None` |

`payload` 只给 `engine_tokens` 使用，表示该位置由 input builder 用外部 embedding、
multimodal feature、latent-derived embedding 或 handle 填充。没有自然 token id 的模态
也必须使用 dummy / placeholder id，保证 position、token budget、`token_ids_cpu` 与
vLLM 兼容。

## 3. Request State

```python
@dataclass
class UADRequestState:
    request_id: str
    phase: UADPhase

    engine_tokens: list[UADToken]
    materialized_tokens: list[UADToken]
    num_computed_tokens: int

    stop_token_ids: set[int]
    phase_switch_token_id: int | None = None
    phase_switch_reason: str | None = None

    dit_step_index: int = 0
    dit_num_steps: int = 0
    dit_query_tokens: int = 0

    active_token_start: int | None = None
    active_token_len: int = 0

    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    generator_state: object | None = None
    scheduler_state: object | None = None
    image_state: object | None = None

    final_output: object | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.engine_tokens)
```

| 变量 | 含义 | 更新方 |
|---|---|---|
| `engine_tokens` | 模型可见 token 序列 | sampler、transition hook、final DiT step |
| `materialized_tokens` | 用户可见输出序列 | sampler、artifact epilogue |
| `num_tokens` | `len(engine_tokens)` | 派生值 |
| `num_computed_tokens` | `engine_tokens` 中已提交到可复用 engine state 的前缀长度 | prefill/decode forward、final DiT commit |
| `dit_step_index` | 已完成 DiT step 数 | DiT output handler |
| `dit_num_steps` | 当前 DiT 总 step 数 | transition hook |
| `dit_query_tokens` | 一个 DiT step 的 token budget | transition hook |
| `active_token_start/active_token_len` | 当前生成 image/chunk 在 `engine_tokens` 中的位置 | transition hook |
| `latents/timesteps/*_state` | DiT runtime state，不是 scheduler 进度 | transition hook、DiT output handler |

`num_computed_tokens` 不是输出长度，也不是 forward 次数。它只表示：

```text
engine_tokens[:num_computed_tokens]     已提交，可复用
engine_tokens[num_computed_tokens:]     尚未提交
```

原 vLLM 也不是输出与 computed 同步：prompt prefill 推进 computed 但不输出 token；
sampled token 会先进入 request/output ledger，通常到下一次 decode forward 后才被
computed 覆盖。

### 状态更新时间

| 事件 | `engine_tokens` | `materialized_tokens` | `num_computed_tokens` | DiT state |
|---|---|---|---|---|
| 请求进入系统 | 初始化 prompt/context | 空或历史输出 | prefix cache 命中长度或 0 | 初始化为 0/None |
| AR prefill forward | 不变 | 不变 | `+= num_scheduled_tokens` | 不变 |
| AR decode forward | 不变 | 不变 | `+= 1` | 不变 |
| sampler 产出普通 text token | append text token | append text token | 不变 | 不变 |
| sampler 产出 phase switch token | append switch token | 按 API 需要 append | 不变 | transition hook 执行 |
| AR -> DiT transition | append 缺失的 image/chunk placeholder/query tokens；必要时 append end control/EOS | 不变 | 不变 | 设置 active token range、latents、timesteps、step counters、`phase=DIT_STEP` |
| 非 final DiT step | 不变 | 不变 | 不变 | 更新 latents/image state；`dit_step_index += 1` |
| final DiT step | 可 append 尚未存在的 end control/EOS | 可 append control/EOS 输出 | 推进到 active tokens/end control 之后 | commit active context；`dit_step_index += 1` |
| VAE/artifact epilogue | 不变 | append image artifact | 不变 | 设置 `final_output`，进入 `FINISHED` |

原生 multiturn 要求 final DiT step 默认 commit generated image context。VAE decode 只
materialize artifact，不推进 `num_computed_tokens`。

final DiT step 的 computed 增量：

```python
commit_until = active_token_start + active_token_len
if end_control_or_eos_token_is_in_engine_tokens:
    commit_until += num_end_control_tokens
num_new_computed_tokens = commit_until - state.num_computed_tokens
```

## 4. Phase Update

runner/output handler 对 request 的修改只通过 `UADPhaseUpdate` 表达。

```python
@dataclass
class UADPhaseUpdate:
    new_engine_tokens: list[UADToken] = field(default_factory=list)
    new_materialized_tokens: list[UADToken] = field(default_factory=list)
    num_new_computed_tokens: int = 0
```

应用顺序：

```python
state.engine_tokens.extend(update.new_engine_tokens)
write [t.token_id for t in update.new_engine_tokens] to runner token ledger

state.materialized_tokens.extend(update.new_materialized_tokens)

state.num_computed_tokens += update.num_new_computed_tokens
request.num_computed_tokens += update.num_new_computed_tokens
```

约束：`new_engine_tokens[*].token_id` 必须非 None；`new_materialized_tokens` 永远不反馈
到模型输入；`num_new_computed_tokens` 只由 forward 完成和 cache/state commit 决定。

## 5. Scheduler

UAD scheduler 保留 vLLM 主输出，并额外给 runner 一组 per-request item。

```python
@dataclass
class UADScheduleBounds:
    req_id: str
    phase: UADPhase
    min_scheduled_tokens: int
    max_scheduled_tokens: int

@dataclass
class UADRunnerSpec:
    input_kind: Literal[
        "ar_prefill",
        "ar_decode",
        "dit_full_context_step",
        "dit_iter_step",
    ]
    output_kind: Literal["sample_tokens", "denoise_pred", "none"]
    step_index: int | None = None
    total_steps: int | None = None
    shape_bucket: tuple[int, ...] | None = None

@dataclass
class UADScheduleItem:
    req_id: str
    phase: UADPhase
    num_scheduled_tokens: int
    state_policy: Literal["append", "init", "state_update", "read_only", "none"]
    slot_mapping: torch.Tensor | None
    runner_spec: UADRunnerSpec

@dataclass
class UADSchedulerOutput:
    base: SchedulerOutput
    uad_items: list[UADScheduleItem]
```

| 字段 | 创建时机 | 消费方 |
|---|---|---|
| `UADScheduleBounds` | scheduler 选择 request 前 | scheduler budget packing |
| `num_scheduled_tokens` | scheduler 选中 request 后 | `SchedulerOutput`、runner input builder |
| `state_policy` | build schedule item 时 | runner cache/state path |
| `slot_mapping` | block allocator 或 adapter 准备后 | attention/cache metadata builder |
| `runner_spec` | build schedule item 时 | input builder、output handler、DiT patch bucketing |
| `uad_items` | 每个 scheduler tick | runner 组 batch |

规则：

- `num_scheduled_tokens` 是本 tick work size，不是累计进度。
- DiT 长期进度只看 `dit_step_index / dit_num_steps`。
- 剩余 budget 小于 `min_scheduled_tokens` 时，本 tick 跳过该 request。
- 同一个 `UADSchedulerOutput` 可以同时包含 AR 和 DiT items。
- scheduler 不按 attention metadata 预拆队列；attention 分组由 runner/executor 做。

## 6. Runner Inputs

```python
@dataclass
class UADInputs:
    req_ids: list[str]
    items: list[UADScheduleItem]
    input_ids: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    positions: torch.Tensor | None
    latents: torch.Tensor | None
    timesteps: torch.Tensor | None
    attention_metadata: object
    model_kwargs: dict[str, Any]
```

| 字段 | AR | DiT |
|---|---|---|
| `input_ids` | scheduled token ids | placeholder ids，可只作 ledger |
| `inputs_embeds` | 通常为空或 prompt embeds | latent/timestep/image-derived embeddings |
| `latents/timesteps` | 空 | 当前 DiT step 输入 |
| `attention_metadata` | vLLM paged causal metadata | runner 生成的 causal + bidirectional patch plan |
| `model_kwargs` | 原 AR kwargs | 模型私有 DiT tensors/state |

`items` 保留 scheduled rows 到 request/phase 的映射。DiT non-final steps 不 append
denoise-step tokens 到 `token_ids_cpu`；phase switch 不释放 request。

## 7. Executor Attention

executor 不按 phase 拆两个完整 forward。每层执行：

```text
norm / QKV projection over unified hidden buffer
attention core branches
O projection + FFN/MoE over unified hidden buffer
```

MVP attention core：

1. 对本 tick 全部 scheduled tokens 跑 vLLM 原生 causal paged attention。每个 token 都
   causal attend 到自己和前文。AR token 的最终 attention output 就是 causal output。
2. 对 DiT current chunk 额外跑 bidirectional patch。patch 只计算 chunk 内 causal mask
   缺失的非因果边；不同 request chunk 不互相 attend。
3. 对 DiT query，用 causal partial 和 patch partial 做 logsumexp merge。

合并要求：causal kernel 返回 `out_causal/lse_causal`；bidirectional patch 返回
`out_patch/lse_patch`。如果现有 paged attention kernel 不返回 lse，必须扩展 kernel
或重算 causal partial。

```python
m = max(lse_causal, lse_patch)
w_causal = exp(lse_causal - m)
w_patch = exp(lse_patch - m)
out = (w_causal * out_causal + w_patch * out_patch) / (w_causal + w_patch)
```

Bidirectional patch 只在 `input_kind/state_policy/shape_bucket/mask/qkv shape` 兼容时
合 batch。AR token 不进入 bidirectional patch。

## 8. Output

```python
@dataclass
class UADPhaseOutput:
    req_id: str
    phase: UADPhase
    sampled_token_ids: list[int] | None = None
    denoise_pred: torch.Tensor | None = None
    artifact: object | None = None
    update: UADPhaseUpdate = field(default_factory=UADPhaseUpdate)
```

| `output_kind` | 处理 |
|---|---|
| `sample_tokens` | logits -> sampler -> `new_engine_tokens`，通常也产生 `new_materialized_tokens` |
| `denoise_pred` | 不走 sampler；adapter 更新 latents/image state |
| `none` | 内部状态迁移或无输出 |

Artifact epilogue 不占 scheduler budget。HunyuanImage3 的 cache/state commit 在 final
DiT step；VAE decode 只 append image artifact 到 `materialized_tokens`。

## 9. HunyuanImage3 MVP

```text
AR_PREFILL -> AR_DECODE
  -- sampler emits <img_ratio_*>, transition hook
  -> DIT_STEP x N
  -- VAE artifact epilogue
  -> FINISHED
```

Transition hook：

- 解析 image size / ratio。
- append 缺失的 image placeholder/query `engine_tokens`；已 sampled 的
  `<img_ratio_*>` 不重复 append。
- 如 end control/EOS 已确定，可以 append 到 `engine_tokens`，final commit 覆盖它。
- 设置 `active_token_start/active_token_len`。
- 初始化 `latents/timesteps/generator_state/scheduler_state/image_state`。
- 设置 `dit_step_index=0`、`dit_num_steps`、`dit_query_tokens`、`phase=DIT_STEP`。

DiT item：

```python
if state.dit_step_index == 0:
    runner_spec.input_kind = "dit_full_context_step"
    state_policy = "init"
else:
    runner_spec.input_kind = "dit_iter_step"
    state_policy = "state_update"
```

DiT update：

- non-final step：只更新 DiT runtime state，`dit_step_index += 1`。
- final step：commit generated image context，推进 `num_computed_tokens` 到 image
  tokens 和 end control/EOS 之后，`dit_step_index += 1`。
- VAE epilogue：materialize image artifact，不推进 `num_computed_tokens`。

## 10. DreamZero / Chunk Refinement MVP

DreamZero 类模型按“生成 active chunk -> 多步 refine -> commit 成 frozen context ->
继续下一个 chunk”接入，不新增 phase。

```text
AR_PREFILL -> AR_DECODE
  -- chunk spec / boundary token, transition hook
  -> DIT_STEP x K       # refine active chunk
  -- final/noise=0 commit
  -> AR_DECODE or DIT_STEP for next chunk
  ...
  -> artifact epilogue
  -> FINISHED
```

映射：

| 概念 | UAD 表达 |
|---|---|
| frozen chunks | `engine_tokens[:num_computed_tokens]`，只读 prefix context |
| active chunk | `engine_tokens[active_token_start:active_token_start + active_token_len]` |
| chunk sigma/noise schedule | `timesteps/scheduler_state` |
| chunk latent state | `latents/image_state` |
| chunk refine step | `DIT_STEP` item，`num_scheduled_tokens = dit_query_tokens` |
| chunk commit | final `DIT_STEP` 推进 `num_computed_tokens` |

Transition hook：

- 解析下一个 chunk 的大小、位置和条件。
- append active chunk placeholder/query `engine_tokens`。
- 设置 `active_token_start/active_token_len`。
- 初始化该 chunk 的 `latents/timesteps/scheduler_state/image_state`。
- 设置 `dit_step_index=0`、`dit_num_steps=chunk_sigma_steps`、
  `dit_query_tokens=active_token_len`、`phase=DIT_STEP`。

DiT update：

- non-final refine step：只更新 chunk runtime state，`dit_step_index += 1`。
- final/noise=0 step：把 clean chunk state commit 到 reusable engine state，
  `num_computed_tokens` 推进到 active chunk 之后，`dit_step_index += 1`。
- 如果还有下一个 chunk，`phase` 回到 `AR_DECODE` 或直接进入下一个 chunk 的
  transition；如果没有，artifact epilogue materialize 最终输出。

Computed 增量：

```python
commit_until = active_token_start + active_token_len
num_new_computed_tokens = commit_until - state.num_computed_tokens
```

注意：refine 的中间 sigma steps 不 append engine tokens，也不推进
`num_computed_tokens`。只有 chunk commit 后，该 chunk 才成为后续 chunk 可见的 frozen
context。

## 11. Adapter 接口

```python
class UADModelAdapter(Protocol):
    def init_request_state(self, request) -> UADRequestState: ...
    def get_schedule_bounds(self, state, budget) -> UADScheduleBounds | None: ...
    def build_schedule_item(self, state, n) -> UADScheduleItem: ...
    def build_inputs(self, items, runner_state) -> UADInputs: ...
    def forward(self, inputs, forward_context) -> Any: ...
    def process_output(self, item, raw, state) -> UADPhaseOutput: ...
```

## 12. TODO: CFG Parallel

CFG parallel 不进入 MVP。后续需要明确：

- cond/uncond 是同一个 request 下的两个 branch，还是两个 scheduler-visible child
  items。
- `engine_tokens`、prefix KV、active chunk state 在两个 branch 间如何共享。
- `num_scheduled_tokens` 如何计费：按 logical tokens 计一次，还是按 branch 数放大。
- `runner_spec` 是否需要表达 `cfg_branch_id`、`cfg_group_id` 和 merge policy。
- attention executor 是把 CFG branches 当普通 batch rows 跑，还是在 branch 维度做
  专用 packing。
- CFG merge 在 denoise output handler 中完成；merge 后只更新一次 latents/image
  state。
- final commit 只能发生一次，不能让 cond/uncond branch 分别推进
  `num_computed_tokens`。

## 13. TODO: SP / Ring / Ulysses

SP 不进入 MVP。后续需要明确：

- SP 只作用于 DiT bidirectional patch，还是也改造 causal base。
- active chunk 按 sequence 维切分后，`active_token_start/active_token_len` 与每个 rank
  的 local token range 如何映射。
- Ring / Ulysses 的通信发生在 attention core 内；projection/FFN/MoE 是否仍按 TP/DP
  现有路径执行。
- causal base + bidirectional patch 的 logsumexp merge 在 SP 下如何合并跨 rank
  `lse`。
- attention metadata 需要携带 local q/k/v range、global chunk range、SP group、
  padding/ragged 信息。
- scheduler 是否只按 global `dit_query_tokens` 计费，runner 再拆 local tokens。
- 不同 shape bucket / SP group size 的 DiT requests 是否允许同一 tick 混合。

## 14. MVP 范围

MVP 必须支持 AR-only 兼容、同 tick AR/DiT 调度、HunyuanImage3 单 engine
AR -> DiT -> VAE、DreamZero 类 chunk refinement、attention core 分支下的
projection/FFN/MoE 合 batch、以及 multiturn image/chunk context commit。

MVP 暂不支持 CFG parallel、sequence parallel / Ring / Ulysses、完整 paged mixed-mask
DiT attention、强制 AR+DiT attention 单 kernel 混跑、跨节点 KV transfer 重设计。
