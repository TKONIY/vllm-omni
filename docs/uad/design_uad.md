# UAD 统一引擎设计

目标：基于 vLLM AR engine 做一个 unified AR + DiT inference engine。第一版采用
类似 nano-vLLM 的最小闭环：少量对象、清晰状态流、先跑通 research path，再逐步替换成
vLLM production scheduler / KV manager / runner 能力。

第一目标模型是 HunyuanImage3。当前只讨论 DP + TP；CFG parallel 和 SP 只保留 TODO。

## 1. Nano-UAD 核心闭环

主循环保持一个方向：

```text
UADEngine.step()
  -> UADScheduler.schedule()
  -> UADRunner.execute_model(scheduler_output)
  -> UADScheduler.update_from_output(scheduler_output, runner_output)
       -> HunyuanImage3UADStateMachine.update_request_state(...)
  -> serving/output_processor materialize
```

核心对象只有五类：

| 对象 | 职责 |
|---|---|
| `UADRequestState` | 单 request 的 engine ledger、AR sampler ledger、phase、DiT metadata |
| `UADScheduler` | 持有 requests，按 phase 产出 work items，应用 runner output |
| `UADRunner` | 把 scheduled items pack 成 batch，执行 model forward/logits/sample |
| `HunyuanImage3UADModel` | 模型 forward shell；不更新 request state |
| `HunyuanImage3UADStateMachine` | 模型私有 token/phase 规则，例如 `<img_ratio_*>` |

对齐 vLLM 的原则：

- scheduler 统一调度多个 request，不让 runner 读 request 对象。
- runner 不识别 Hunyuan 私有 token，不调用 state machine。
- state machine 只返回 request delta，不执行模型。
- serving/output processor 只处理 materialized text / artifact，不参与 token budget。

## 2. Request State

```python
UADPhase = Literal["ar_prefill", "ar_decode", "dit_step"]

@dataclass
class UADRequestState:
    request_id: str
    engine_tokens: list[UADToken]
    materialized_tokens: list[UADToken]
    ar_sampler_token_ids: list[int]
    phase: UADPhase
    num_computed_tokens: int

    dit_step_index: int
    total_dit_steps: int
    image_context_token_count: int
    pending_image_context_commit: bool
```

字段语义：

- `engine_tokens`：模型后续可见的统一多模态 context。
- `materialized_tokens`：用户可见 text token；后续 image artifact 也走 materialized/output
  path，不走 AR sampler ledger。
- `ar_sampler_token_ids`：只用于构造 vLLM `SamplingMetadata.output_token_ids` 的 AR
  generated-token history。它不是 `engine_tokens[prompt_len:]`，也不是用户输出。
- `num_computed_tokens`：`engine_tokens` 中已经提交到 reusable engine state / KV 的前缀
  长度。它不是 forward 次数，也不是 materialized 输出长度。
- `phase`：scheduler 根据它决定本 tick 生成哪类 work item。

## 3. Scheduler

`UADScheduler` 是 nano-style 单一 scheduler。它负责：

- `add_request()` / `get_request()`。
- `schedule()`：把 runnable requests 转成 `UADScheduleItem`。
- `update_from_output()`：消费 runner raw output，调用 state machine，应用 delta。

```python
@dataclass
class UADScheduleItem:
    request_id: str
    phase: UADPhase
    token_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    persist: bool
    dit_step_index: int | None
    total_dit_steps: int | None
    ar_sampler_token_ids: list[int]
    sample_token_offset: int | None
```

关键字段：

- `phase`：runner 执行 recipe。当前只有 `ar_prefill` / `ar_decode` / `dit_step`。
- `num_scheduled_tokens`：本 item 占用的 scheduler token budget。
- `num_computed_tokens`：调度时 request 已经提交的 context 前缀长度。
- `persist`：是否把本 item 结果提交为 reusable engine context / KV。
- `ar_sampler_token_ids`：AR sampler history snapshot。
- `sample_token_offset`：本 item 内哪个 hidden state 要投影到 logits；当前 AR MVP 是最后
  一个 scheduled token，对齐 vLLM `logits_indices` 的角色。

当前 toy scheduler 没有复杂 token budget；后续接 vLLM scheduler 时，`UADScheduleItem`
仍是 runner 看到的最小 contract。

## 4. KV Lifecycle

不能直接照搬 nano-vLLM 的纯 AR block manager。UAD 必须从第一版保留 phase-aware KV
语义：

| Work item | `persist` | KV 行为 | `num_computed_tokens` |
|---|---:|---|---|
| AR prefill | `True` | 分配 slot，写 paged KV | 推进 |
| AR decode | `True` | 分配 slot，写 paged KV | 推进 |
| DiT non-final denoise | `False` | 不分配 writable paged KV，只用 scratch/dense state | 不推进 |
| DiT final image-context commit | `True` | 分配 slot，写 image context KV | 推进 |
| VAE/artifact decode | N/A | 不写 KV，不消耗 token budget | 不推进 |

第一版实现可以没有完整 KV manager，但接口方向必须保留：

```python
class UADKVManager:
    def can_allocate(item: UADScheduleItem) -> bool: ...
    def allocate_persist_slots(item: UADScheduleItem) -> KVMetadata: ...
    def build_block_table(request: UADRequestState) -> ...: ...
    def build_slot_mapping(item: UADScheduleItem) -> ...: ...
    def free(request: UADRequestState) -> None: ...
```

最终实现应是 vLLM `KVCacheManager` 的薄包装，而不是自研 production eviction/prefix-cache。
prefix cache、preemption、eviction 可以后置；`persist=True/False`、block table、slot
mapping 的语义不能后置。

## 5. Runner

`UADRunner` 消费完整 `UADSchedulerOutput`，构造 `UADBatchInputs`，调用模型 forward，
再做 logits/sampling/denoise 后处理。

```python
class UADRunner:
    def execute_model(self, scheduler_output: UADSchedulerOutput) -> UADModelRunnerOutput: ...
```

runner 可以：

- pack AR / DiT items 成一个 batch。
- 根据 `phase` 选择输入 recipe。
- 在 model forward 后执行 AR `compute_logits()` / `sample()`。
- 为后续 attention 构造 block table / slot mapping / mask / position metadata。
- 返回 raw `UADModelRunnerOutput`，保持 item 顺序可 scatter。

runner 不可以：

- 判断 `<img_ratio_*>` 是否切 phase。
- 决定 token 是否 materialize。
- 修改 `UADRequestState`。
- 调用 state machine。

## 6. Batch Contract

```python
@dataclass
class UADBatchItem:
    request_id: str
    phase: UADPhase
    output_index: int
    num_tokens: int
    token_start: int
    token_end: int
    num_computed_tokens: int
    persist: bool
    dit_step_index: int | None
    total_dit_steps: int | None
    ar_sampler_token_ids: tuple[int, ...]
    sample_token_offset: int | None

@dataclass
class UADBatchInputs:
    items: tuple[UADBatchItem, ...]
    input_token_ids: torch.Tensor
    token_item_indices: torch.Tensor
    token_positions: torch.Tensor
```

`phase` 是唯一 recipe selector：

- AR phase：`input_token_ids` 是真实 token IDs。
- DiT phase：当前 toy path 使用 fake latent slots；真实路径会换成 latent/timestep inputs。

## 7. HunyuanImage3 MVP

当前 Milestone B 是 AR backend API smoke path，不是完整真实 AR path：

- `HunyuanImage3UADModel` 只调用 backend `forward(input_ids, positions)`。
- `UADRunner` 调 backend `compute_logits()` 和 `sample()`。
- `SamplingMetadata` 只覆盖 greedy/no-penalty 和 vLLM `output_token_ids`。
- 真实 backend path 显式限制单 AR item；多请求 batched AR 要等 paged KV / attention
  metadata 接入。

当前 toy DiT：

- sampled `<img_ratio_*>` 后，state machine append toy image context tokens。
- non-final `dit_step` 使用 `persist=False`，只推进 DiT step index。
- final `dit_step` 使用 `persist=True`，把 pending image context 标记为 computed。
- 不做真实 denoise / VAE / artifact。

## 8. State Machine

每个模型定义自己的 state machine：

```python
class UADModelStateMachine:
    def update_request_state(
        self,
        request: UADRequestState,
        runner_output: UADModelRunnerItemOutput,
    ) -> UADStateUpdate: ...
```

`UADStateUpdate` 是 request delta：

```python
@dataclass
class UADStateUpdate:
    request_id: str
    new_engine_tokens: list[UADToken]
    new_materialized_tokens: list[UADToken]
    new_ar_sampler_token_ids: list[int]
    phase_update: UADPhaseUpdate | None
    finished: bool
```

HunyuanImage3 state machine 当前负责：

- 识别 `<img_ratio_*>`。
- 判断 control token 是否 engine-only。
- 生成 toy image context tokens。
- 设置 DiT metadata：image size、latent shape、step count、seed、guidance scale。
- 推进 toy DiT step。

`new_ar_sampler_token_ids` 只更新 AR sampler history，不承载 text/image 对外输出。

## 9. 后续 Milestones

优先级从 correctness 到性能：

1. 接 vLLM paged KV metadata：block table、slot mapping、`persist=True/False` slot lifecycle。
2. 单请求真实 AR：在 `set_forward_context()` 下跑 HunyuanImage3 AR forward/logits/sample。
3. 单请求真实 DiT：prefix paged attention + chunk dense bidirectional attention + LSE merge。
4. VAE/artifact output：image artifact 走 materialized/output path。
5. continuous batching：多个 request 同 tick，先 phase 分组正确，再做 AR/DiT layer 合批。
6. AR + DiT FFN/MoE 合批：attention recipe 可以不同，FFN/MoE hidden batch 必须合并。

## 10. TODO

CFG parallel：

- CFG branch 表达。
- branch cache 共享。
- denoise merge。
- final commit 语义。

SP：

- ring attention / Ulysses 的选择。
- SP 与 paged-prefix attention 接口。
- SP + EP 的 all-to-all 和 token routing。
