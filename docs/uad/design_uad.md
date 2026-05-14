# UAD 统一引擎设计

目标：基于 vLLM AR engine 做一个统一 AR + DiT engine。一个 scheduler tick 可以同时
调度 AR request 和 DiT request，runner 消费同一个 batch，从而给 projection/FFN/MoE
更大的有效 token batch。

第一目标模型是 HunyuanImage3。MVP 只讨论 DP + TP；CFG parallel 和 SP 只列 TODO。

## 1. 核心原则

- 复用 vLLM v1 的 request lifecycle、scheduler、paged KV、block table、slot mapping、
  continuous batching 和 output path。
- scheduler 仍按 token budget 做调度；不同 phase 只是影响一个 request 本 tick 能调度
  什么 work item。
- runner 不识别模型私有 token，不调用模型状态机。runner 只根据 scheduler output 构造
  batch、执行统一的 HunyuanImage3UADModel、返回 raw output。
- 模型私有规则由 state machine 表达，例如 HunyuanImage3 的 `<img_ratio_*>`、control
  token、AR -> DiT phase switch。
- request state 的更新发生在 scheduler 的 `update_from_output()`，对齐 vLLM 原本
  `Scheduler.update_from_output(SchedulerOutput, ModelRunnerOutput)`。

## 2. 与 vLLM 原路径的对应

| vLLM AR engine | UAD engine |
|---|---|
| `Request` | `UADRequestState` |
| `Scheduler.schedule()` | `UADScheduler.schedule()` |
| `SchedulerOutput` | `UADSchedulerOutput` |
| `GPUModelRunner.execute_model(scheduler_output)` | `UADRunner.execute_model(uad_scheduler_output)` |
| 具体 `nn.Module` model | `HunyuanImage3UADModel.forward(uad_batch_inputs)` |
| `ModelRunnerOutput.sampled_token_ids` | `UADRunnerOutput.sampled_token` / denoise raw output |
| `Scheduler.update_from_output(...)` | `UADScheduler.update_from_output(...)` |
| `EngineCoreOutput(new_token_ids=...)` | `UADStepOutput(new_engine_tokens, new_materialized_tokens, ...)` |
| serving `OutputProcessor` / detokenizer | UAD materialized text/artifact output processor |

主循环必须保持这个形状：

```text
scheduler.schedule()
  -> runner.execute_model(scheduler_output)
  -> scheduler.update_from_output(scheduler_output, runner_output)
       -> state_machine.update_request_state(...)
       -> UADStepOutput / EngineCoreOutput
  -> serving/output_processor materialize
```

这意味着：

- scheduler 统一选择多个 request 的 work item。
- runner 消费完整 `UADSchedulerOutput`，可以按 phase 分组执行，但不能逐 request 调
  state machine。
- state machine 只把 raw runner output 翻译成 request state delta；它不触发 runner。
- serving/output processor 只处理已经 materialized 的输出，不参与 scheduler token budget。

## 3. Request State

```python
UADPhase = Literal["ar_prefill", "ar_decode", "dit_step"]

@dataclass
class UADRequestState:
    request_id: str
    engine_tokens: list[UADToken]
    materialized_tokens: list[UADToken]
    phase: UADPhase
    num_computed_tokens: int

    dit_step_index: int
    total_dit_steps: int
    image_context_token_count: int
    pending_image_context_commit: bool
```

关键语义：

- `engine_tokens`：模型后续可见的统一多模态 context。
- `materialized_tokens`：用户可见输出，用于 streaming/detokenize/artifact。
- `num_computed_tokens`：`engine_tokens` 中已经提交到 engine reusable state 的前缀长度。
  它不是 forward 次数，也不是 materialized 输出长度。
- `phase`：scheduler 用它决定本 tick 允许调度的 work item。

原 vLLM 里 prompt prefill 会推进 `num_computed_tokens` 但不输出 token；sampled token
会先进入 request/output ledger，通常下一次 decode forward 后才变成 computed。UAD 保持
这个分离。

## 4. Scheduler

UAD scheduler 负责：

- 持有 `UADRequestState`。
- 根据 request phase 和 token budget 生成 `UADSchedulerOutput`。
- 对需要写入 paged KV 的 item，复用 vLLM `KVCacheManager` 分配 blocks，并把 block ids
  放进 scheduler output。
- 在 `update_from_output()` 中消费 runner raw output，调用 model-specific state machine，
  更新 request state，并产出 `UADStepOutput` / 后续 `EngineCoreOutput`。

当前 toy 结构：

```python
@dataclass
class UADScheduleItem:
    request_id: str
    phase: UADPhase
    token_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    persist: bool

class UADScheduler:
    def add_request(...) -> UADRequestState: ...
    def schedule() -> UADSchedulerOutput: ...
    def update_from_output(
        self,
        scheduler_output: UADSchedulerOutput,
        runner_output: UADRunnerStepOutput,
    ) -> UADStepOutput: ...
```

字段语义：

- `request_id`：本 item 属于哪条 request。
- `phase`：runner 执行路径，当前是 `ar_prefill` / `ar_decode` / `dit_step`。
- `token_ids`：token-backed AR work 的输入 token ids。DiT denoise 的输入来自
  latent/timestep state，可以为空。
- `num_scheduled_tokens`：scheduler token budget 里本 item 占用的 token 数。
- `num_computed_tokens`：调度时 request 已经写入 reusable context/KV 的
  `engine_tokens` 前缀长度。
- `persist`：context commit bit。`persist=True` 表示本 item 成功执行后必须写入可复用
  engine context/KV，并推进 `num_computed_tokens`；`persist=False` 表示 transient compute，
  不推进 `num_computed_tokens`。

例子：

| Work item | `persist` | 结果 |
|---|---:|---|
| AR prefill | `True` | prompt/context tokens 写 paged KV，推进 `num_computed_tokens` |
| AR decode | `True` | 本轮 scheduled token 写 paged KV，推进 `num_computed_tokens` |
| DiT non-final denoise | `False` | 只更新 diffusion runtime state，不推进 `num_computed_tokens` |
| DiT final image-context write | `True` | image context 写 paged KV，推进 `num_computed_tokens` |

注意：`persist` 是 engine context 语义；`materialized_tokens` 是 serving/output 语义。
前者决定后续 token 能否 attend 到该 context，后者决定是否对用户 streaming /
detokenize / 返回 artifact。

当前 MVP 的 token 规则：

| Phase | scheduled item | `num_scheduled_tokens` | `persist` |
|---|---|---:|---|
| `ar_prefill` | prompt/context chunk | `len(pending_tokens)` toy；后续接 chunked prefill | `True` |
| `ar_decode` | 上一步 sampled token | `1` toy | `True` |
| `dit_step` | 一个 denoise step | `image_context_token_count` toy | non-final `False`，final `True` |

真实实现里，DiT final `dit_step(persist=True)` 必须通过 vLLM
`KVCacheManager.allocate_slots()` 分配 slots 并写 paged KV。non-final
`dit_step(persist=False)` 只消耗 UAD work budget，不推进 `num_computed_tokens`。

## 5. Runner

`UADRunner` 是 batch-first 的执行编排层，负责把 mixed AR + DiT items 打成一个
layer-wise batch，再调用 `HunyuanImage3UADModel`：

```python
class UADRunner:
    def execute_model(self, scheduler_output: UADSchedulerOutput) -> UADRunnerStepOutput: ...
```

runner 可以：

- pack mixed AR + DiT items 成 `UADBatchInputs`。
- 为 attention 构造 phase / mask / position metadata。
- 为 DiT 构造 latents / timesteps / image-shape metadata。
- 让 AR + DiT hidden tokens 进入同一个 FFN / MoE batch。
- 保持 item 顺序可还原，并返回 raw `UADRunnerOutput`。

runner 不可以：

- 判断 `<img_ratio_*>` 是否是 phase switch。
- 决定 token 是否应该 materialize。
- 修改 `UADRequestState`。
- 调用 state machine。

当前 toy shell 已经固定如下执行契约：

```python
# vllm_omni/uad/batch.py

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
    input_kind: Literal["token_ids", "latent_timestep"]
    dit_step_index: int | None
    total_dit_steps: int | None

@dataclass
class UADBatchInputs:
    items: tuple[UADBatchItem, ...]
    input_token_ids: torch.Tensor
    token_item_indices: torch.Tensor
    token_positions: torch.Tensor
```

AR item 的 `input_token_ids` 是真实 token id；DiT item 当前用 fake latent slots
占位，真实实现会替换成 latent/timestep recipe。`output_index` 是 runner scatter
回 scheduler item 顺序的唯一依据。这些通用 batch dataclass 不放在任何具体模型文件里。

batch contract 不单独定义 attention 类型。runner/model 直接根据 `phase` 选择
执行 recipe：AR phase 走 AR causal paged path；DiT phase 走 DiT prefix paged
attention（读取 prefix，`causal=False`）和 chunk-local bidirectional attention，
两路结果后续用 LSE merge。

### HunyuanImage3UADModel

`HunyuanImage3UADModel` 位于 `vllm_omni/uad/model/hunyuan_image3.py`，是一个
research-only 的单一 `nn.Module` shell。当前不注册到 `model_executor`。

- 它拥有共享 backbone、attention、FFN/MoE、norm、RoPE、token embedding、timestep
  embedding 和权重加载逻辑。
- 它消费 runner 构造的 batch 输入，不消费 request state。
- AR 只是 token 输入 recipe；DiT 只是 latent + timestep 输入 recipe。
- 它返回 raw logits / sampled-token 辅助信息 / denoise prediction，不做 request 更新。
- 它不认识 `persist` / `materialized_tokens` / scheduler token budget。
- 它的职责是让同一个 layer pass 里，AR 和 DiT token 都能参与 attention 和 FFN/MoE 批处理。

当前 Step 4 版本只做 fake compute：AR 输出仍然是最后一个 token `+1`，DiT 输出只表示
scheduled item 完成；但 model forward 已经是 batch-first，一次 forward 可以同时覆盖
AR 和 DiT item。

## 6. State Machine

每个模型定义自己的 state machine。

```python
class UADModelStateMachine:
    def update_request_state(
        self,
        request: UADRequestState,
        runner_output: UADRunnerOutput,
    ) -> UADModelOutput: ...
```

通用基类在 `vllm_omni/uad/state/base.py`。具体模型只继承该基类并实现
`update_request_state()`；HunyuanImage3 的实现位于
`vllm_omni/uad/state/hunyuan_image3.py`。

HunyuanImage3 state machine 当前负责：

- 识别 `<img_ratio_*>`。
- 判断 Hunyuan control token 是否只进入 engine ledger。
- 在 AR -> DiT 时 append toy image context tokens。
- 设置 `phase="dit_step"`、`dit_step_index`、`total_dit_steps`。
- 对 fake DiT step 只推进 `dit_step_index`。
- 它不负责 batch packing，也不负责 model forward。

state machine 返回的是 request state delta：

```python
@dataclass
class UADModelOutput:
    request_id: str
    new_engine_tokens: list[UADToken]
    new_materialized_tokens: list[UADToken]
    phase_update: UADPhaseUpdate | None
    finished: bool
```

`UADScheduler.update_from_output()` 负责应用这个 delta。`num_computed_tokens` 不由
state machine 直接返回；它由 scheduler 根据对应 `UADScheduleItem.persist` 和
`num_scheduled_tokens` 推进。

## 7. HunyuanImage3 MVP

当前 toy 行为：

```text
AR output token:
  state_machine.update_request_state(...)
  if ordinary text:
      new_engine_tokens += token
      new_materialized_tokens += token
  if <img_ratio_*>:
      new_engine_tokens += [ratio_token, toy_img_tokens..., optional_eoi]
      new_materialized_tokens unchanged
      phase = dit_step

DiT step raw output:
  state_machine.update_request_state(...)
  dit_step_index += 1
  non-final step: persist=False, no engine token commit
  final step: persist=True, pending image context becomes computed
  no materialized artifact
```

后续真实 HunyuanImage3 接入时：

- AR phase 走同一个 `HunyuanImage3UADModel` 的 token recipe。
- DiT phase 走同一个 `HunyuanImage3UADModel` 的 latent + timestep recipe。
- UADRunner 在同一个 batch 里同时 pack AR / DiT items，并把 FFN/MoE 合成一个 shared batch。
- DiT non-final step 使用 dense scratch K/V，不写 vLLM paged KV。
- DiT 读取 text/prefix context 时使用 read-only paged-prefix attention。
- DiT final `dit_step(persist=True)` 默认写 vLLM paged KV，支持原生 multiturn。
- VAE/artifact decode 只 materialize image artifact，不推进 `num_computed_tokens`。

## 8. Attention 与 KV

第一版 attention 目标：

- 所有 token 对历史 context 的 causal/prefix attention 尽量复用 vLLM paged attention。
- DiT chunk 内 full attention 使用 dense K/V。
- DiT prefix paged attention 与 chunk dense attention 通过 LSE merge 合并。

KV 约束：

- scheduler 负责决定哪些 token 会写 paged KV，并在 forward 前完成 slot allocation。
- runner 只消费 scheduler output 中的 block/slot metadata。
- `persist=False` 的 DiT denoise step 不写 paged KV。
- final `dit_step(persist=True)` 必须写 paged KV，避免 multiturn 只能靠外部 artifact。

## 9. TODO

CFG parallel：

- 表达 CFG branch。
- branch cache 共享。
- denoise merge。
- final commit 语义。

SP：

- DiT sequence parallel 的 ring / Ulysses 选择。
- SP 与 paged-prefix attention 的接口。
- SP + EP 的 token routing 和 all-to-all 开销。

真实 serving output：

- `UADStepOutput` 对齐 `EngineCoreOutput`。
- text streaming 与 image artifact materialization 分离。
- VAE/artifact decode 作为 serving output epilogue，而不是 scheduler phase。
