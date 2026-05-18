# UAD Implementation Plan

目标：用接近 nano-vLLM 的最小对象集合搭一个 research-only UAD engine，但保留
vLLM/UAD 必须有的 KV 生命周期语义：`persist`、`block_table`、`slot_mapping`、以及
DiT scratch compute 和 final image-context commit 的区别。

当前开发位置：

```text
branch: uad-code
worktree: ~/code/vllm-omni-uad-code
```

第一个完整 HunyuanImage3 运行边界：

```text
prompt
  -> HunyuanImage3 AR forward / logits / sample
  -> HunyuanImage3 state machine 识别 image boundary
  -> DiT denoise loop
  -> final image context persist 到 paged KV
  -> VAE decode
  -> image artifact output
```

首个完整运行版本不做 CFG parallel、SP、真实吞吐优化、motivation benchmark。

## 1. 当前已完成

已完成的是最小控制面和 AR backend smoke path，不是完整真实 HunyuanImage3。

- `UADRequestState`：统一记录 engine context、materialized output、AR sampler history、
  phase、DiT metadata、`num_computed_tokens`。
- `UADScheduler`：持有 request，`schedule()` 产出 work item，
  `update_from_output()` 调 state machine 并应用 request delta。
- `UADRunner`：消费完整 `UADSchedulerOutput`，pack mixed AR/DiT batch，调用 model shell，
  再做 AR logits/sample。
- `HunyuanImage3UADModel`：UAD-local model shell；当前可用 toy path 或单 AR backend
  smoke path。
- `HunyuanImage3UADStateMachine`：模型私有 token/phase 规则；runner 不识别
  `<img_ratio_*>`。
- `UADScheduleItem.persist`：`True` 表示执行成功后推进 reusable context/KV 和
  `num_computed_tokens`；non-final DiT step 为 `False`。

已验证：

- toy AR / DiT phase switch / final persist 单测。
- HunyuanImage3 state metadata 单测。
- fake Hunyuan-style AR backend smoke 单测。
- runner 不调用 state machine；state machine 不执行模型。

## 2. 简化原则

从 nano-vLLM 借鉴的部分：

- 对象少：request、scheduler、runner、model、state machine。
- scheduler 直接持有 request，先不拆复杂生产调度器。
- runner 输入输出都是小 dataclass，便于单测和替换。

不能照搬 nano-vLLM 的部分：

- 不能用纯 AR block manager 语义。UAD 必须区分 `persist=True/False`。
- DiT non-final step 是 computed but not persisted，不写 paged KV，也不推进
  `num_computed_tokens`。
- final DiT step 要把 image context 写成后续 multiturn 可读的 reusable context。
- `block_table` / `slot_mapping` 最终必须来自 vLLM `KVCacheManager`，不是另写
  production page manager。

## 3. Milestone C：Paged KV / Persist 接口

难度：L。目标：不新增 page manager，先把 UAD schedule item 对接 vLLM paged KV
metadata 的最小路径。

实现步骤：

1. 给 `UADSchedulerOutput` 增加 runner 所需 KV metadata 占位：
   `block_table`、`slot_mapping`、per-item computed/scheduled token 信息。
2. 引入 UAD-local KV wrapper，只做 vLLM `KVCacheManager` 的薄封装。
3. `persist=True` item 分配 writable slots；`persist=False` item 不分配 writable slots。
4. AR prefill/decode 写 paged KV 后推进 `num_computed_tokens`。
5. final DiT commit 写 image-context KV 后推进 `num_computed_tokens`。
6. non-final DiT step 只保留 scratch/dense state，不推进 context。

验证：

- AR prefill/decode 的 `num_computed_tokens` 与 vLLM 语义一致。
- image context 跨多个 KV page 时 block allocation 正确。
- `persist=False` 不分配 writable slots、不推进 computed。
- final DiT 后同一 request 能继续下一轮 AR decode。
- request finish 后 blocks 释放，无泄漏。

停点：提交并等待 review。

## 4. Milestone D：真实 AR Path，单请求优先

难度：L。目标：在 UAD runner 下跑 HunyuanImage3 AR 的真实 forward/logits/sample。

实现步骤：

1. 复用现有 HunyuanImage3 AR model module，不复制模型结构。
2. 在 `set_forward_context()` 下传入 UAD 生成的 positions、KV metadata、attention
   metadata。
3. 保持职责对齐 vLLM：model forward 只产 hidden states；runner 调
   `compute_logits()` 和 `sample()`。
4. 继续用 `ar_sampler_token_ids` 构造 vLLM `SamplingMetadata.output_token_ids`。
5. 第一版限制单 request，先不承诺真实 AR batch。

验证：

- fake backend 单测继续通过。
- 小模型或 mock 权重下 AR forward/logits/sample smoke 通过。
- sampled `<img_ratio_*>` 仍由 state machine 切到 DiT。
- AR sampler history 不包含 image/context placeholder。

停点：提交并等待 review。

## 5. Milestone E：真实 DiT Denoise，单请求

难度：XL。目标：单请求跑通真实 DiT denoise loop，不做 CFG/SP。

实现步骤：

1. 复用现有 HunyuanImage3 DiT layer、timestep embedding、postprocessor。
2. DiT runtime state 放在 request state 或 runner-side state：
   latents、timesteps、shape bucket、seed、guidance scale。
3. DiT attention recipe：
   - DiT query 读已 persist prefix：paged attention，`causal=False`。
   - DiT chunk 内：dense bidirectional attention。
   - 两路 attention 用 LSE merge。
4. non-final DiT step：更新 latent，`persist=False`，不写 KV。
5. final DiT step：生成 image context embedding/tokens，走 Milestone C 的
   `persist=True` KV commit。

验证：

- 单 DiT layer / 单 timestep 与现有 diffusion pipeline 做数值对齐。
- fixed prompt/seed 下 latent deterministic。
- non-final step 不推进 `num_computed_tokens`。
- final step 推进 `num_computed_tokens`，并保留 multiturn context。

停点：提交并等待 review。

## 6. Milestone F：VAE Decode / Artifact Output

难度：M。目标：从 UAD engine 返回真实图片 artifact。

实现步骤：

1. 复用现有 HunyuanImage3 VAE / image processor。
2. VAE decode 不作为 scheduler phase，不消耗 token budget，不写 KV。
3. `UADEngineCoreOutputs` 或 serving 层增加 image artifact 输出结构。
4. output processor 区分 text token、engine-only control token、image artifact。

验证：

- 单 prompt 输出 PNG/PIL smoke。
- image artifact output 不改变 `num_computed_tokens`。
- text-only request 行为不变。

停点：这是第一个完整 HunyuanImage3 E2E；提交并等待 review。

## 7. Milestone G：Continuous Batching Correctness

难度：L。目标：一个 scheduler tick 可同时调度多个 request，混合 AR item 和 DiT item。

实现步骤：

1. 在现有 `UADScheduler` 上补 token budget、waiting/running、finished cleanup。
2. 仍按 token 数计费：AR 是 scheduled token，DiT step 是 image-context token count。
3. runner 按 phase 构造 attention recipe，但保持统一 item scatter 顺序。
4. 先保证 correctness，再考虑性能。

验证：

- 多 AR request 连续 decode。
- AR + DiT 同 tick 调度。
- 长 DiT request 不饿死新 AR request。
- 多请求结果与单请求顺序执行一致。
- finish 后 KV blocks 释放正确。

停点：提交并等待 review。

## 8. Milestone H：AR + DiT Layer 合批

难度：XL。目标：AR 和 DiT 的 attention recipe 可以不同，但 projection/FFN/MoE 要吃同一个
hidden token batch。

实现步骤：

1. runner/model 建立统一 hidden buffer，记录 item offset 和 phase。
2. attention 分 recipe：
   - AR：causal paged attention。
   - DiT：prefix paged non-causal attention + chunk dense bidirectional attention + LSE merge。
3. attention output 写回统一 hidden buffer。
4. projection / FFN / MoE 在统一 buffer 上执行。
5. 输出按 item scatter 回 AR logits 或 DiT denoise output。
6. 增加 trace：每层 FFN/MoE token batch size、per-phase token count、EP expert token
   histogram。

验证：

- mixed execution 与 phase-separated execution 数值一致或在容差内。
- trace 能看到 AR + DiT token 合并后的 FFN/MoE batch。
- 单请求和多请求都能跑。

停点：提交并等待 review。

## 9. Motivation Experiments

这个不阻塞 E2E，使用现有 staged HunyuanImage3 serving，不动 UAD engine。

Online serving sweep：

- 从高到低 request rate 打流量。
- 记录 AR/DiT busy interval、stage idle ratio、每个 forward 的 FFN/MoE token batch size、
  latency、queue wait、error。

FFN/MoE saturation microbench：

- 单独测 HunyuanImage3 一个 FFN/MoE 层。
- TP 和 EP 配置下递增 token 数。
- 记录 latency、tokens/s、achieved TFLOPs、EP local expert token histogram、饱和阈值。

最终输出：把 online trace 的 token 分布叠到 FFN/MoE 饱和阈值上，证明 UAD 合批动机。

## 10. TODO：CFG / SP / Production

CFG parallel：

- branch 表达、branch cache 共享、denoise merge、final commit 语义。

SP：

- ring attention / Ulysses 选择。
- SP 与 paged-prefix attention 接口。
- SP + EP 的 all-to-all 和 token routing。

Production serving：

- OpenAI-compatible output schema。
- artifact streaming。
- cancellation / timeout / retry。
- production model registry、TP/EP/quant loader 集成。
