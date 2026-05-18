# UAD Milestone Plan

目标：基于 vLLM AR engine 新开 `UADEngine`，让 HunyuanImage3 可以在同一套
request / scheduler / runner / KV 管理下完整运行 AR + DiT，并逐步支持 continuous
batching 和 AR/DiT 合批。

当前分支和 worktree：

```text
branch: uad-code
worktree: ~/code/vllm-omni-uad-code
```

## 0. Milestone 定义

“完整运行 HunyuanImage3”的最小验收边界：

```text
prompt
  -> real HunyuanImage3 AR prefill/decode
  -> real HunyuanImage3 state machine 识别 image boundary
  -> real DiT denoise loop
  -> final image context persist 到 engine context / paged KV
  -> VAE decode
  -> request output 返回 image artifact
```

不纳入第一个完整运行 milestone 的内容：

- CFG parallel。
- SP / ring attention / Ulysses。
- AR + DiT 真实 FFN/MoE 混合合批。
- 多请求吞吐优化。
- motivation benchmark。

难度标记：

| 标记 | 含义 |
|---|---|
| S | 小改动，主要是接口/测试/文档 |
| M | 需要读现有实现并补齐状态或数据结构 |
| L | 接入真实模型路径或 vLLM 关键基础设施 |
| XL | attention / distributed / 性能关键路径，风险高 |

## 1. 已完成：Toy Control Plane

难度：S。状态：完成。

已实现：

- `UADEngine` / `UADToyScheduler` / `UADRunner` / `UADRequestState`。
- `UADScheduleItem.persist`，`persist=True` 推进 `num_computed_tokens`。
- `UADModelRunnerOutput` / `UADStateUpdate` / `UADEngineCoreOutputs`。
- HunyuanImage3 toy state machine：toy ratio token 触发 AR -> DiT。
- toy `HunyuanImage3UADModel`：runner 可以 pack mixed AR + DiT item，一次 forward 后 scatter。

已验证：

- UAD 单测 23 个通过。
- runner 不识别 HunyuanImage3 私有 token。
- state machine 只改 request state，不调用 runner。
- scheduler `update_from_output()` 是唯一 request state update 路径。

## 2. Milestone A：真实 HunyuanImage3 State / Metadata

难度：M。状态：完成。

目标：去掉 toy token 规则，让 UAD state machine 与现有 HunyuanImage3 AR -> DiT
边界一致，但仍不跑真实大模型。

实现步骤：

1. 从现有 tokenizer / config 构造 `HunyuanImage3UADStateConfig`。
2. 对齐现有 `vllm_omni/model_executor/stage_input_processors/hunyuan_image3.py`：
   - image ratio token。
   - image size / bucket。
   - image token count。
   - AR -> DiT 所需 metadata。
3. 补齐 HunyuanImage3 私有 token 规则：
   - think / recaption / answer / boi / eoi。
   - engine-only token。
   - 对外 materialize 的 text token。
4. 增加 `UADRequestState` 中真实 DiT 需要的轻量 metadata：
   - image size。
   - latent shape。
   - timestep count。
   - seed。
5. 保持 runner 语义不变：runner 仍不读取 tokenizer/token 规则。

验证：

- 用 tokenizer fixture 测所有特殊 token id 解析。
- 用固定 token 序列测试 state transition：
  `ar_decode -> dit_step -> ar_decode/finished`。
- 与 `stage_input_processors/hunyuan_image3.py` 的 AR -> DiT metadata 输出做 parity test。
- 不加载真实模型，只跑 CPU unit tests。

停点：完成后提交，等待 review。

## 3. Milestone B：AR Backend API Smoke Path，单请求

难度：L。状态：完成。

目标：UAD 可以把单个 AR item 交给现有 HunyuanImage3-style AR backend，按 vLLM
runner 职责拆分执行 `forward -> compute_logits -> sample`，直到产生普通 text token
或 image boundary token。没有 paged KV / attention metadata 前，它不是完整真实 AR
path。

已实现：

1. `HunyuanImage3UADModel` 保留 toy fallback；传入 `ar_model` 时只调用 backend
   `forward(input_ids, positions)`。
2. `UADRunner` 负责 backend `compute_logits(last_hidden_state)` 和
   `sample(logits, SamplingMetadata)`；不可用时回退到 argmax。
3. `UADRunner` 显式携带 `sample_token_offset`，对应 vLLM `logits_indices` 的单 item
   MVP 语义。
4. `UADRequestState.ar_sampler_token_ids` 单独维护 AR sampler history，不从
   `engine_tokens` 推导，避免把 image/context placeholder 传给 sampler；它不表示
   用户可见 text/image 输出。
5. 真实 backend path 显式限制单 AR request；多请求真实 batch 留给后续 milestone。
6. 当前 `SamplingMetadata` 只支持 greedy/no-penalty smoke；temperature/top-p、
   repetition penalty、logprobs、logit bias 仍未接入。
7. `HunyuanImage3UADStateMachine` 继续消费 sampled token，决定 text continuation 或
   AR -> DiT transition。

验证：

- fake Hunyuan-style backend 验证 UAD 按 runner 职责调用 `forward / compute_logits / sample`。
- prefill/decode positions 与 token ids 传递正确。
- `ar_sampler_token_ids` 只包含 AR sampled token，不包含 image context placeholder。
- 多 AR item 使用真实 backend 时会报错，避免伪装成 batched vLLM path。
- sampled ratio token 仍由 state machine 切到 `dit_step`。
- 不加载真实权重；真实 loader、paged KV、TP/EP 仍留到后续 milestone。

停点：完成后提交，等待 review。

## 4. Milestone C：Paged KV / Persist 接入

难度：L。目标：UAD 不新增 page manager，所有 reusable context 都复用 vLLM paged KV。

实现步骤：

1. 让 `UADSchedulerOutput` 携带 vLLM runner 需要的 KV metadata：
   - block table。
   - slot mapping。
   - num computed / scheduled tokens。
2. `UADScheduler` 复用 vLLM `KVCacheManager` 为 `persist=True` item 分配 slots。
3. AR prefill/decode：
   - `persist=True`。
   - forward 写 paged KV。
   - 成功后推进 `num_computed_tokens`。
4. DiT non-final denoise：
   - `persist=False`。
   - 不分配 writable slots。
   - 不推进 `num_computed_tokens`。
5. final DiT：
   - `persist=True`。
   - 为 generated image context tokens 分配 slots。
   - 写入 paged KV。
   - 推进 `num_computed_tokens` 到 image context 之后。

验证：

- page boundary case：image context token 数跨多个 KV page。
- block allocation / free 无泄漏。
- AR prefill/decode 的 `num_computed_tokens` 与 vLLM 原语义一致。
- final DiT 后同一 request 的下一轮 text 能接着调度。
- 暂不要求真实 DiT attention；可以用 controlled fake image context 写 KV 验证 persist。

停点：完成后提交，等待 review。

## 5. Milestone D：真实 DiT Attention + Denoise，单请求

难度：XL。目标：单请求完整跑真实 DiT denoise loop，不做 CFG parallel / SP。

实现步骤：

1. 引入真实 DiT runtime state：
   - latents。
   - timesteps。
   - image grid / shape bucket。
   - per-step scheduler state。
2. `HunyuanImage3UADModel` 增加真实 DiT branch：
   - 复用 `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`。
   - 复用 timestep embedding、DiT decoder layers、postprocessor。
3. 实现 DiT attention recipe：
   - DiT Q 读历史 prefix：paged-prefix attention，`causal=False`。
   - DiT chunk 内：dense bidirectional attention。
   - 两路 attention 用 LSE merge。
4. non-final DiT step：
   - 更新 latent / denoise state。
   - `persist=False`。
   - 不写 paged KV。
5. final DiT step：
   - 生成 image context embedding / tokens。
   - 通过 Milestone C 的 persist 路径写 paged KV。
6. 第一版固定：
   - 单请求。
   - 固定 image size bucket。
   - `guidance_scale <= 1.0`，不启用 CFG。

验证：

- 单个 DiT layer / 单个 timestep 与现有 diffusion pipeline 的输出做数值对齐。
- 完整 denoise loop 无 NaN。
- fixed prompt / seed 下 latent deterministic。
- non-final step 不推进 `num_computed_tokens`。
- final step 推进 `num_computed_tokens` 并保留 multiturn context。

停点：完成后提交，等待 review。

## 6. Milestone E：VAE Decode / Artifact Output

难度：M。目标：从 UAD engine 返回真实图片 artifact，形成第一个完整 HunyuanImage3 E2E。

实现步骤：

1. 复用现有 HunyuanImage3 VAE / image processor。
2. DiT denoise 完成后触发 VAE decode。
3. VAE decode 不作为 scheduler phase，不消耗 token budget，不写 KV。
4. `UADEngineCoreOutputs` 增加 image artifact 输出所需字段，或增加 serving 层转换结构。
5. output processor：
   - text token streaming。
   - engine-only control token 不输出。
   - image artifact 最终返回。

验证：

- 单 prompt -> PNG / PIL image smoke test。
- request 正常 finished。
- generated image context 已 persist，artifact decode 不改变 `num_computed_tokens`。
- text-only request 行为不被破坏。

停点：这是第一个“完整运行 HunyuanImage3” milestone。完成后提交，等待 review。

## 7. Milestone F：Continuous Batching Correctness

难度：L。目标：一个 scheduler tick 可以同时包含多个 request，且可以混合 AR item 和 DiT item。

实现步骤：

1. 将 toy scheduler 换成基于 vLLM scheduler 语义的 UAD scheduler：
   - token budget。
   - waiting / running request。
   - preemption / finished cleanup 的最小版本。
2. schedule item 仍用 token 数计费：
   - AR prefill/decode 是 scheduled tokens。
   - DiT step 是 image context token count。
3. runner 按 phase 分组执行：
   - AR group。
   - DiT group。
4. 保持 request output 顺序可还原。
5. 保证 state machine 仍只在 scheduler `update_from_output()` 中调用。

验证：

- 多个 AR request 连续 decode。
- AR + DiT request 同 tick 调度。
- 长 DiT request 不饿死新 AR request。
- 多请求结果与单请求顺序执行结果一致。
- request finish 后 KV blocks 正确释放。

停点：完成后提交，等待 review。

## 8. Milestone G：真实 AR + DiT Layer 合批

难度：XL。目标：attention recipe 可以分开，但 projection / FFN / MoE 必须能把 AR token 和
DiT token 合成更大的 batch。

实现步骤：

1. runner 建立统一 hidden buffer：
   - item span。
   - phase。
   - output scatter index。
2. attention 先按 recipe 执行：
   - AR causal paged attention。
   - DiT prefix paged + chunk dense attention。
3. attention output 写回统一 hidden buffer。
4. 共享 projection / FFN / MoE 在统一 hidden buffer 上执行。
5. FFN/MoE 输出按 span 切回 AR logits 或 DiT denoise output。
6. 加最小 trace：
   - 每层 FFN/MoE local token batch size。
   - per-phase token count。

验证：

- 固定 seed 下，mixed FFN/MoE 与 phase-separated 执行数值一致或在容差内。
- request state 更新顺序不变。
- FFN/MoE batch size trace 能看到 AR + DiT token 合并。
- 单请求和多请求都能跑。

停点：完成后提交，等待 review。

## 9. Milestone H：Distributed / Production Integration

难度：XL。目标：让 UAD 路径能在实际 HunyuanImage3 部署配置上运行。

实现步骤：

1. 明确 UAD model 是否注册到 production model registry，或保持 research-only entrypoint。
2. 接 vLLM model loader：
   - TP shard。
   - quant。
   - weight tying / shared module。
3. 接 MoE runtime：
   - TP。
   - EP。
   - expert routing stats。
4. 支持最小线上配置：
   - 单组 4 GPU。
   - 先 TP 或 TP+EP。
   - CFG/SP 暂不纳入。
5. 增加 UAD serve / offline smoke script。

验证：

- 4 GPU 上加载真实 HunyuanImage3。
- 单请求 E2E image generation。
- 多请求 smoke。
- 显存释放正常。
- 与非 UAD staged serving 的基础输出语义一致。

停点：完成后提交，等待 review。

## 10. Motivation Experiments

难度：M-L。目标：证明 UAD 的性能动机。这个可以和 Milestone F/G 并行，不阻塞第一个完整
运行 milestone。

### Online Serving Sweep

使用现有 HunyuanImage3 staged online serving，不使用 UAD engine。从高到低 request rate
打流量，记录：

- AR stage / DiT stage busy interval。
- stage idle ratio。
- 每个 forward 的 FFN/MoE token batch size。
- request latency / queue wait / error。

### FFN/MoE Saturation Microbench

单独测 HunyuanImage3 一个 FFN/MoE 层，在 TP 和 EP 配置下递增 token 数，记录：

- latency vs tokens。
- tokens/s vs tokens。
- achieved TFLOPs vs tokens。
- EP local expert token histogram。
- 饱和阈值。

最终输出：把 online trace 的 token 分布叠到 FFN/MoE 饱和阈值上。

## 11. 暂不做但必须保留接口位置

CFG parallel：

- CFG branch 表达。
- CFG branch cache 共享。
- denoise merge。
- final persist 语义。

SP：

- ring attention / Ulysses 的选择。
- SP 与 paged-prefix attention 接口。
- SP + EP 的 all-to-all 和 token routing。

Production serving：

- OpenAI compatible output schema。
- artifact streaming。
- request cancellation / timeout / retry。
