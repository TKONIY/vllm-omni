# UAD 实验与实现计划

本文只保留当前 UAD 方向：基于 vLLM AR engine 新开 `UADEngine`，控制流对齐
`schedule -> runner -> scheduler.update_from_output -> serving/output_processor`。

## 0. 当前状态

工作分支和 worktree：

```text
branch: uad-code
worktree: ~/code/vllm-omni-uad-code
```

已完成：

| Step | 状态 | 说明 |
|---|---|---|
| Step 0 | 完成 | toy `UADEngine`、request ledger、HunyuanImage3 UAD toy model |
| Step 1 | 完成 | `UADScheduleItem` / `UADSchedulerOutput` |
| Step 2 | 完成 | HunyuanImage3 toy AR -> DiT phase switch |
| Step 3 | 完成 | runner 消费完整 scheduler output；state update 移到 scheduler `update_from_output()` |
| Step 5 foundation | 完成 | toy final `dit_step(persist=True)` 推进 pending engine context |

下一步：Step 4 接真实 HunyuanImage3 DiT 单请求路径。

## 1. Motivation 实验计划

目标：证明 staged HunyuanImage3 online serving 里 AR/DiT 两组 GPU 会出现互补空闲，
并证明小 token batch 下 FFN/MoE 不饱和。

### 1.1 Online Serving Sweep

部署现有 HunyuanImage3 online serving，不使用 UAD engine。使用合理 prompt 数据集，从高到低
request rate 打流量，记录每个 stage 的 forward 时间线和 FFN/MoE token batch。

核心输出：

- 每个 request rate 下 AR stage / DiT stage 的 busy interval。
- 每个 request rate 下 stage idle ratio。
- 每个 forward 中 FFN/MoE local token batch size 随时间变化曲线。
- request latency、queue wait、错误样本。

### 1.2 FFN/MoE Saturation Microbench

单独测 HunyuanImage3 一个 FFN/MoE 层，在 TP 和 EP 配置下递增 token 数。

核心输出：

- latency vs tokens。
- tokens/s vs tokens。
- achieved TFLOPs vs tokens。
- EP 下 local expert token histogram。
- 饱和阈值：吞吐达到 plateau 90% 且连续点稳定。

最终把 online trace 的 FFN/MoE token 分布叠到 microbench 饱和阈值上。

## 2. UAD 控制流

必须对齐 vLLM 原结构：

```text
scheduler.schedule()
  -> runner.execute_model(scheduler_output)
  -> scheduler.update_from_output(scheduler_output, runner_output)
       -> state_machine.update_request_state(...)
       -> UADStepOutput / EngineCoreOutput
  -> serving/output_processor materialize
```

职责边界：

| 组件 | 职责 | 不做 |
|---|---|---|
| `UADScheduler` | 持有 request；统一 schedule 多 request；update request state | 不执行模型 |
| `UADRunner` | 消费完整 scheduler output；按 phase 分组执行；返回 raw output | 不识别模型私有 token；不改 request |
| `UADModelStateMachine` | 模型私有 phase/output-ledger 规则 | 不调用 runner；不调度；不决定 computed 进度 |
| serving/output processor | materialize text/artifact | 不参与 token budget |

## 3. 模块边界

```text
vllm_omni/uad/
  request.py          # UADRequestState / UADPhase / UADToken
  scheduler.py        # UADSchedulerOutput / UADToyScheduler.update_from_output()
  runner.py           # UADRunner.execute_model()
  state_machine.py    # UADModelStateMachine protocol
  outputs.py          # UADRunnerOutput / UADModelOutput / UADStepOutput
  engine.py           # UADEngine shell
  omni/
    hunyuan_image3.py # HunyuanImage3 state machine and token rules

vllm_omni/model_executor/models/hunyuan_image3/
  hunyuan_image3_uad.py
```

`HunyuanImage3UADForConditionalGeneration` 放在原 HunyuanImage3 model 目录下，方便后续接
loader、TP、quant 和原 model executor 约定。

## 4. 已实现 Toy 语义

### Step 0/1

- `UADEngine.add_request()` 创建 `UADRequestState`。
- `UADToyScheduler.schedule()` 根据 request phase 生成 `UADScheduleItem`。
- `UADRunner.execute_model()` 消费完整 `UADSchedulerOutput`。
- AR toy model 对最后一个 token 做 `+1`，返回 sampled token。

### Step 2/3

- `HunyuanImage3UADStateMachine` 识别 `<img_ratio_*>`。
- ratio token 触发 `phase="dit_step"`。
- toy image context tokens 进入 `engine_tokens`，不进入 `materialized_tokens`。
- non-final toy `dit_step(persist=False)` 只推进 `dit_step_index`。
- final toy `dit_step(persist=True)` 推进所有 pending engine tokens 到
  `num_computed_tokens`，然后回到 `ar_decode`。
- request state 更新现在在 scheduler `update_from_output()` 中完成。

验证点：

- text token 同时进入 `engine_tokens` 和 `materialized_tokens`。
- Hunyuan control token 只进入 `engine_tokens`。
- runner 不持有 state machine。
- runner 不识别 HunyuanImage3 ratio/control token。
- 一个 scheduler tick 可以同时包含 AR 和 DiT request。

## 5. Paged KV 约束

UAD 不新增 page manager。凡是会写入 reusable paged KV 的 token，都必须复用 vLLM
`KVCacheManager`、block table 和 slot mapping。

| Phase | KV 行为 |
|---|---|
| AR prefill/decode | `persist=True`；写 vLLM paged KV；推进 `num_computed_tokens` |
| DiT non-final denoise | `persist=False`；不写 paged KV；只更新 diffusion runtime state |
| DiT final denoise | `persist=True`；写 image context 的 vLLM paged KV；推进到 image context 后 |
| VAE/artifact decode | 不写 paged KV；只 materialize artifact |

DiT 读取历史 text/prefix context 时，目标路径是 read-only paged-prefix attention。
DiT chunk 内 full attention 使用 dense K/V，两路 attention 通过 LSE merge 合并。

## 6. Step 4：真实 HunyuanImage3 DiT 单请求

目标：把 fake DiT step 替换成真实 HunyuanImage3 DiT 单请求路径。

范围：

- 复用 `HunyuanImage3Model` 的 DiT block、time embedding、final layer。
- 根据 AR 产出的 size/ratio 初始化 latents 和 timesteps。
- 一次 UAD `dit_step` item 对应一个 denoise timestep。
- 先只支持单请求、固定 shape、固定 steps。
- 先不做 CFG parallel、SP、mixed FFN/MoE。

验证：

- 单请求从 AR ratio token 进入真实 DiT denoise。
- 每个 DiT step 由 scheduler 调度，由 runner 执行，由 scheduler update state。
- non-final DiT step 不改 `num_computed_tokens`。
- final `dit_step(persist=True)` 设计明确，暂未强制完成真实 paged KV 写入。

## 7. Step 5：Final DiT `persist=True`

目标：让 final DiT step 默认把 generated image context 写入 vLLM paged KV，并通过
`persist=True` 推进 `num_computed_tokens`，原生支持
multiturn。

范围：

- `UADScheduleItem` 明确定义 `persist`：`True` 表示本 item 成功后写 reusable context/KV
  并推进 `num_computed_tokens`；`False` 表示 transient compute。
- scheduler 将 non-final `dit_step` 标成 `persist=False`，final `dit_step` 标成
  `persist=True`。
- toy runner 先 mock final persist 完成；真实 runner 后续根据 HunyuanImage3 image grid
  生成 image context embeddings / positions，并写入 paged KV。
- 真实路径需要在 forward 前用 vLLM `KVCacheManager` 分配 image tokens 的 slots。
- scheduler 根据 final `dit_step(persist=True)` 推进 `num_computed_tokens` 到 image context
  之后。
- VAE/artifact decode 仍不进入 scheduler token budget。

验证：

- final `dit_step(persist=True)` 后 `num_computed_tokens` 覆盖 image context tokens。
- 下一轮 text turn 可以接在同一条 `engine_tokens` 后继续调度。
- 当前 toy 验证不覆盖 block table / slot mapping；真实 vLLM KV 接入后必须验证无 block 泄漏。

## 8. Step 6：Paged-prefix attention + dense chunk attention

目标：让 DiT denoise token 读取已提交的 text/image prefix，同时保留 chunk 内 bidirectional
attention。

范围：

- DiT Q 对历史 prefix 走 read-only paged-prefix attention，`causal=False`。
- DiT chunk 内 K/V 使用 dense scratch buffer 做 full attention。
- 两路 attention output 用 LSE merge 合并。
- 先支持单请求固定 shape，不实现 CFG parallel / SP。

验证：

- DiT denoise step 读取 vLLM paged prefix，不复制整段 prefix KV 到长期 dense cache。
- attention output shape、dtype、device 与 DiT block 预期一致。
- 固定 prompt / seed 下完整 denoise 无 NaN，step state 正常推进。

## 9. Step 7：多请求 phase-separated batching

目标：一个 scheduler tick 可以同时调度多个 AR 和 DiT request，runner 先按 phase 分组执行。

范围：

- scheduler 在同一个 token budget 下选择 AR prefill/decode 和 DiT step item。
- runner 分组执行 AR group 与 DiT group，保持 request output 顺序可还原。
- 不做 mixed FFN/MoE；不做 CFG parallel / SP。

验证：

- 多个请求并发时，AR 和 DiT item 能在同一个 tick 被调度。
- phase-separated 执行结果与单请求顺序执行结果一致。
- 长 DiT request 不饿死新进 AR request，AR decode 不饿死 DiT step。

## 10. Step 8：Mixed projection / FFN / MoE batch

目标：在 attention 边界仍分 phase 的前提下，把可共享的 projection/FFN/MoE token 合成更大
batch。

范围：

- runner 建立统一 hidden buffer，记录每个 item 的 token span。
- attention 各自执行后，把 AR/DiT hidden states 拼到共享 projection/FFN/MoE。
- FFN/MoE 输出再按 span 切回各 request/phase。
- 先不改 attention kernel，不引入 SP。

验证：

- 固定 seed 下 mixed FFN/MoE 与 phase-separated 执行的 logits/denoise output 在容差内一致。
- FFN/MoE trace 中 local token batch 分布明显右移。
- request 输出顺序和 request state 更新不受 mixed batch 影响。

## 11. Step 9：Serving output processor

目标：把 UAD 的 `materialized_tokens` / image artifact 对齐到 serving output path。

范围：

- 将 `UADStepOutput` 映射到后续 `EngineCoreOutput` / request output。
- text token 支持 delta streaming。
- VAE/artifact decode 作为 output epilogue materialize image，不作为 scheduler phase。
- engine-only control/image context tokens 不对外 streaming。

验证：

- text-only request streaming 行为与普通 AR 路径一致。
- image request 最终返回 image artifact，且不会把 Hunyuan control tokens 暴露给用户。
- final-only 与 delta output mode 均能正常结束并释放 request。

## 12. TODO

CFG parallel：

- CFG branch 表达。
- CFG branch cache 共享。
- denoise merge。
- final commit 语义。

SP：

- ring attention / Ulysses 的选择。
- SP 与 paged-prefix attention 接口。
- SP + EP 的 all-to-all 和 token routing。

Motivation 实验整理：

- online serving sweep 脚本。
- FFN/MoE saturation microbench。
- trace summary 和 plotting。
