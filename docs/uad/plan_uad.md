# UAD 实验与实现计划

目标分两部分：

1. 用现有 HunyuanImage3 online serving 证明 staged AR + DiT serving 的 FFN/MoE
   batch 不稳定且容易空闲，并量化 FFN/MoE 需要多少 token 才能跑满。
2. 在不改动原 AR engine 主路径的前提下，新开 `UADEngine`，逐步复用 vLLM AR
   scheduler、request ledger、paged KV、runner input batch 和 output path。

实验脚本统一放在 `docs/uad/script/`。在线 serving sweep 的入口是
`docs/uad/script/run_hunyuan_motivation_experiment.sh`，它会生成 TP+TP / TP+EP
stage config、启动现有 HunyuanImage3 staged serving、打 request-rate sweep，并把
metrics / logs / summary 写到 `artifacts/uad_motivation/<run_id>/`。
FFN trace 由 `UAD_TRACE_FFN=1` 打开，汇总脚本是
`docs/uad/script/summarize_uad_ffn_trace.py`。FFN 饱和 microbenchmark 脚本是
`docs/uad/script/bench_hunyuan_ffn_saturation.py`，汇总脚本是
`docs/uad/script/summarize_ffn_saturation.py`。

## 0. 当前状态

代码进度：

| 项 | 状态 | 说明 |
|---|---|---|
| Step 0：Toy HunyuanImage3 UAD 入口 | 已完成并推送 | commit `62c4bb08` |
| Step 1：UAD scheduler shadow item | 已完成并推送 | commit `0ddbf3b6` |
| Step 2：HunyuanImage3 toy phase switch | 已完成并推送 | commit `375086c3` |
| Step 3：runner-first toy DiT step | 已完成 | runner 直接执行 toy AR/DiT item，不再通过独立翻译层 |
| 下一步 | Step 4 | 接入真实 HunyuanImage3 DiT 单请求路径 |

当前实现工作区使用独立 worktree：

```text
~/code/vllm-omni-uad-code
```

这个 worktree 只保留已提交的 UAD 实现线，不混入 motivation 实验临时改动。motivation
实验如果继续做，应在单独 step 中整理脚本和 trace hook，避免污染 engine 实现审核面。

此前 motivation 实验涉及的文件类型包括：

- `vllm_omni/utils/uad_trace.py`：FFN/MoE JSONL tracer。
- `vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py`：AR MoE tracing hook。
- `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`：DiT MoE tracing hook。
- `benchmarks/diffusion/diffusion_benchmark_serving.py`：metrics 中补充 request rate、shape、
  steps、error samples。
- `vllm_omni/entrypoints/openai/serving_chat.py`：只给 GLM-Image 注入
  `target_h/target_w`，避免 HunyuanImage3 的 mRoPE signature 被错误 extra args 打断。
- `docs/uad/script/`：motivation sweep、stage config 生成、FFN/MoE saturation 和汇总脚本。
- `artifacts/uad_motivation/`：本地实验输出，不应随实现 step 提交。

这些 motivation 改动不属于 Step 2/3。除非专门整理 motivation 实验，否则实现线继续从
Step 4 的真实 HunyuanImage3 DiT 单请求路径开始。

## 1. Motivation 实验计划

### 1.1 要验证的问题

实验要回答两个问题。

**问题 A：现有 two-stage serving 是否有空闲。**

现状 HunyuanImage3 是 AR stage 生成中间结果，再交给 DiT stage denoise/VAE。两个
stage 分别拥有 GPU 组。需要验证在不同 request rate 下：

- AR stage 是否等待 DiT stage 或 downstream queue。
- DiT stage 是否等待 AR stage 产出。
- 两组 GPU 是否出现明显互补空闲。
- forward 内 FFN/MoE 的 token batch 是否长期小于饱和阈值。

**问题 B：HunyuanImage3 FFN/MoE 需要多少 token 才能跑满。**

单独测一个 FFN/MoE 层，在不同并行配置下递增 token 数，得到：

- latency vs tokens。
- tokens/s vs tokens。
- achieved TFLOPs vs tokens。
- local expert token histogram vs tokens。
- 饱和 token 阈值：吞吐达到 plateau 的 90% 且连续 3 个点稳定。

### 1.2 统一指标定义

后续所有实验都写 JSONL trace。每行是一条事件。

```json
{
  "ts_ns": 0,
  "rank": 0,
  "stage_id": 0,
  "stage_type": "ar",
  "req_id": "request-id",
  "phase": "ar_prefill|ar_decode|dit_step|vae_decode",
  "event": "forward_start|forward_end|ffn_start|ffn_end|moe_route",
  "layer_id": 0,
  "local_tokens": 0,
  "global_tokens_est": 0,
  "hidden_size": 0,
  "tp_size": 1,
  "ep_size": 1,
  "sp_size": 1,
  "local_expert_tokens": [0],
  "duration_us": 0
}
```

核心派生指标：

| 指标 | 定义 | 用途 |
|---|---|---|
| `stage_busy_interval` | 同一 rank 上 `forward_start -> forward_end` | 算 stage idle ratio |
| `stage_idle_ratio` | `1 - busy_time / wall_time` | 判断两组 GPU 是否空闲 |
| `ffn_local_tokens` | 当前 rank 输入 FFN/MoE 的 token 数 | 画 FFN batch size 时间曲线 |
| `ffn_global_tokens_est` | `local_tokens * tp_size * sp_size`，仅作估算 | 跨并行配置对比 |
| `local_expert_tokens` | EP 下本 rank 各 local expert 收到的 token 数 | 判断 MoE 是否因 token 少而不饱和 |
| `queue_wait_ms` | request arrival 到 stage forward start | 判断 stage 间阻塞 |

AR 的 `local_tokens` 来自 vLLM runner 中 flattened hidden states 的第一维。DiT 的
`local_tokens` 来自 DiT hidden states 的 `batch * local_seq_len`。开 SP 时记录 local
sequence tokens；画图时同时展示 local 和 estimated global tokens。

### 1.3 Online serving workload

主 workload 使用 text-to-image，固定输出尺寸和步数，避免 image size / step 数混杂。

| 项 | 设置 |
|---|---|
| 模型 | `tencent/HunyuanImage-3.0-Instruct` |
| serving | 现有 staged online serving |
| 推荐配置 | `docs/uad/hunyuan_image3_stage_moe_debugpy.sh`，`DEBUGPY_WAIT_FOR_CLIENT=0` |
| 输出尺寸 | `1024x1024` |
| diffusion steps | `50` |
| prompt 集 | 首选 GEBench type3/type4 或 PartiPrompts 子集；没有数据时用 `benchmarks/diffusion/diffusion_benchmark_serving.py --dataset random` |
| request 数 | pilot 50，正式每个 rate 200 或运行 15 min |
| arrival | Poisson arrival，按 request rate 控制 |

prompt 选择原则：

- 不混合 image editing / image understanding，先只测 text-to-image。
- prompt 长度覆盖短、中、长三档，但固定 image size 和 steps。
- 至少 200 条，避免单个 prompt 的 AR token 长度偶然性。

### 1.4 Request rate sweep

先做 pilot，估计当前配置的稳定吞吐 `qps_base`：

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --backend vllm-omni \
  --dataset random \
  --task t2i \
  --num-prompts 50 \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --request-rate inf \
  --max-concurrency 8 \
  --port 8091
```

正式实验从高到低打流量：

```text
1.50 * qps_base
1.25 * qps_base
1.00 * qps_base
0.75 * qps_base
0.50 * qps_base
0.25 * qps_base
```

每个 rate 单独启动一次 benchmark，保留独立 trace 目录：

```bash
UAD_TRACE_FFN=1 \
UAD_TRACE_DIR=artifacts/uad_motivation/rate_${RATE} \
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --backend vllm-omni \
  --dataset random \
  --task t2i \
  --num-prompts 200 \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --request-rate "${RATE}" \
  --max-concurrency 32 \
  --port 8091
```

`--max-concurrency` 只作为上限，真实 arrival 由 `--request-rate` 控制。

### 1.5 Stage idle 实验 instrumentation

最小改动：

1. 新增轻量 tracer：
   - `vllm_omni/utils/uad_trace.py`
   - 环境变量：`UAD_TRACE_FFN=1`、`UAD_TRACE_DIR=...`、`UAD_TRACE_LAYERS=0,mid,last|all`
   - 每个 rank 写一个 JSONL，避免跨进程锁。
2. AR stage forward interval：
   - 包在 `GPUARModelRunner.execute_model()` 或现有 `record_function("gpu_model_runner: forward")` 外层。
   - 记录 `stage_id`、`rank`、`req_ids`、`num_scheduled_tokens`。
3. DiT stage forward interval：
   - 包在 `DiffusionModelRunner.execute_model()` 的 `pipeline.forward(req)` 外层。
   - 对 HunyuanImage3 额外记录每个 denoise step 的 `forward_call` interval。
4. Queue / stage handoff：
   - 在 orchestrator `_forward_to_next_stage` 记录 AR output produced timestamp。
   - DiT request 第一次 forward 记录 consumed timestamp。

输出图：

- `stage_busy_timeline_rate_<r>.png`：AR stage 与 DiT stage 的 forward intervals。
- `stage_idle_ratio_rate_<r>.json`：每个 stage/rank 的 idle ratio。
- `stage_queue_wait_rate_<r>.png`：stage handoff wait time 分布。

验收标准：

- 至少能看到每个 rate 下 AR/DiT stage 的 busy/idle 时间线。
- 如果 UAD motivation 成立，应在中低 rate 看到 stage GPU 大片空闲；高 rate 下也应看到
  FFN/MoE token batch 呈现尖峰而不是持续大 batch。

### 1.6 FFN batch size 时间曲线

给 AR 和 DiT 的 FFN/MoE module 注册 forward pre/post hook。

记录位置：

- AR：HunyuanImage3 AR 模型的 MLP/MoE block。输入通常是 `[num_tokens, hidden]`。
- DiT：`HunyuanImage3DecoderLayer` 的 MLP/MoE block。输入通常是 `[batch, seq, hidden]`
  或已经 SP-sharded 的 local sequence。

默认只采 3 层降低开销：

```text
layer 0
middle layer
last layer
```

输出图：

- `ffn_tokens_timeline_stage0_rate_<r>.png`
- `ffn_tokens_timeline_stage1_rate_<r>.png`
- `moe_local_expert_tokens_rate_<r>.png`

每张图横轴是 wall-clock time，纵轴是 `local_tokens`；不同颜色表示 layer 或 stage。

验收标准：

- 每个 request rate 都能画出 FFN/MoE token batch 随时间变化。
- 图里能区分 AR prefill、AR decode、DiT denoise step。
- EP 下能看到 local expert token 分布，而不只看总 token。

### 1.7 单层 FFN/MoE microbenchmark

新增脚本：

```text
docs/uad/script/bench_hunyuan_ffn_saturation.py
```

功能：

1. 使用 HunyuanImage3 的 hidden/intermediate/top-k 形状构造等价 FFN GEMM。
2. 在给定 TP/EP local row 规则下用 `torchrun` 初始化 2 卡进程。
3. 使用 CUDA event 计时，不走 online serving，也不加载整模型权重。
4. token 数递增：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
1024, 2048, 4096, 8192, 16384
```

测试矩阵：

| 配置 | 目的 |
|---|---|
| TP=1, EP=off | 单卡 baseline |
| TP=2, EP=off | 看 dense TP 通信影响 |
| TP=4, EP=off | 看 TP 更细 shard 后的饱和阈值 |
| TP=2, EP=on | 看 MoE expert shard 下的 token 阈值 |
| TP=4, EP=on | 对应现有 DiT TP+EP 配置 |
| TP=2, SP=2, EP=on | 验证 SP+EP 组合下 local tokens 是否过小 |

MoE routing 输入两种：

- synthetic uniform：人为让 token 均匀打到 experts，测硬件上限。
- traced routing：从 online trace 采真实 router distribution，测真实负载。

输出图：

- `ffn_latency_vs_tokens_<config>.png`
- `ffn_tokens_per_sec_<config>.png`
- `moe_expert_occupancy_<config>.png`
- `ffn_saturation_threshold.json`

饱和判据：

```text
threshold = 最小 token 数 N，使 throughput(N) >= 0.9 * max_throughput
            并且 N、2N、4N 三个点无明显回落。
```

最终把 online trace 中的 `ffn_local_tokens` 分布叠到 microbench 阈值线上，证明现有
serving 是否长期低于 FFN/MoE 饱和区间。

## 2. UADEngine 实现计划

实现策略：从第一步就新增 HunyuanImage3 的 UAD-native 模型入口，不把 UAD 写成两个
已有 engine 的外层拼接。新入口先做 toy 路径，只要求能跑通；后续逐步复用现有 AR
和 diffusion 模块。第一版不追求覆盖 HunyuanImage3 所有复杂边界条件，优先证明：
同一个 engine 内可以调度 AR token 和 DiT step，并且有机会把 FFN/MoE batch 合大。

### 2.1 新增模块边界

建议新增目录：

```text
vllm_omni/uad/
  __init__.py
  request.py          # UADRequestState / UADPhase / UADToken
  scheduler.py        # UADScheduleItem / UADSchedulerOutput
  runner.py           # UADRunner: input build / model execute / output process
  state_machine.py    # UADModelStateMachine protocol for model-specific phase/output policy
  outputs.py          # UADModelOutput / UADPhaseUpdate
  engine.py           # UADEngine / AsyncUADEngine
  omni/
    hunyuan_image3.py     # HunyuanImage3 state machine / special-token rules

vllm_omni/model_executor/models/hunyuan_image3/
  hunyuan_image3_uad.py   # HunyuanImage3UADForConditionalGeneration
```

`HunyuanImage3UADForConditionalGeneration` 放在现有 HunyuanImage3 AR model 目录下面：

| 位置 | 优点 | 问题 | 当前选择 |
|---|---|---|---|
| `vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py` | 和现有 `HunyuanImage3ForConditionalGeneration` 同目录，方便复用 AR 权重加载、TP、quant、model executor 约定 | research 阶段会多改一点正式 model 目录 | Step 0 起采用 |
| `vllm_omni/uad/models/hunyuan_image3_uad.py` | 改动集中在 UAD 目录 | 离现有 HunyuanImage3 model 太远，后续接 model loader/TP/quant 更别扭 | 不采用 |

入口先用显式开关，不影响现有 serving：

```bash
vllm-omni serve ... --uad-engine
```

新增的 `HunyuanImage3UADForConditionalGeneration` 是第三个入口：

| 入口 | 职责 |
|---|---|
| `HunyuanImage3ForConditionalGeneration` | 现有 AR/T2T/I2T vLLM 模型入口 |
| `HunyuanImage3Pipeline` / `HunyuanImage3Model` | 现有 diffusion pipeline 和 DiT 模型 |
| `HunyuanImage3UADForConditionalGeneration` | UAD 统一执行入口，逐步复用上面两套模块 |

### 2.1.1 Scheduler 与 paged attention 约束

UAD scheduler 不能只返回“要跑多少 token”。只要某个 item 会写入或读取 paged KV，它就
必须兼容 vLLM 的 KV block allocation / block table / slot mapping 契约。

第一版不新增独立 page manager，也不 fork vLLM paged KV 管理逻辑。UAD 只在 scheduler
上层表达 phase 和 work item；凡是需要 reusable KV 的 token，都必须交给 vLLM 原生
`KVCacheManager` 分配 blocks，并交给原生 runner `InputBatch.block_table` 生成页表和
`slot_mapping`。这样 prefix cache、preemption、block free、KV connector、PP/TP 下的
block id 传播都沿用原实现。

原 vLLM v1 的关键路径是：

| 层 | 原 vLLM 行为 | UAD 约束 |
|---|---|---|
| scheduler | 根据 `request.num_tokens_with_spec - request.num_computed_tokens` 计算 `num_new_tokens`，并受 `token_budget` 限制 | UAD 仍按 token budget 调度，但要区分“会写 paged KV 的 token”和“只做 compute 的 DiT step token” |
| KV manager | scheduler 调原生 `kv_cache_manager.allocate_slots(request, num_new_tokens, ...)`，失败则 preempt 或跳过 | 任何要写 paged KV 的 UAD item，都必须用同一个 vLLM `KVCacheManager` 在 forward 前完成 slot allocation |
| SchedulerOutput | 新请求带 `block_ids`；running/resumed 请求带 `new_block_ids`；同时带 `num_scheduled_tokens` | UAD 不能让没有 KV slots 的 item 伪装成 base `SchedulerOutput.num_scheduled_tokens` |
| scheduler post-update | `request.num_computed_tokens += num_scheduled_tokens` 在 schedule 后立即推进 | 只有已分配 KV slots、会进入 paged attention prefix 的 token 才能推进这个值 |
| runner state update | runner 用 `new_block_ids` 更新 `CachedRequestState.block_ids`，并 `input_batch.block_table.append_row(...)` | UAD runner 必须从 scheduler output 更新 block table，不能在 attention 阶段临时猜页表 |
| runner input prep | runner 根据 `num_computed_tokens_cpu + arange` 生成 positions，再由 block table 计算 `slot_mapping` 并 commit 到 GPU | UAD 的 AR token 和最终要 commit 的 image token 都必须有正确 positions 和 slot mapping |

因此回答两个实现问题：

- scheduler 本身不直接改 GPU 页表；它改的是 KV allocation 状态，并把 block ids 放到
  scheduler output。
- 页表实际在 runner 侧更新：`InputBatch.block_table.add_row/append_row` 更新 CPU block
  table，`commit_block_table()` 拷到 GPU，`compute_slot_mapping()` 生成本 tick token 的
  KV slot，`commit_slot_mapping()` 拷到 GPU。

UAD 第一版按这个规则拆：

- AR prefill/decode：完全走 base paged KV contract。
- DiT non-final denoise step：只消耗 UAD work budget，不进入 base
  `SchedulerOutput.num_scheduled_tokens`，不推进 `num_computed_tokens`，不改 block table。
- DiT final image-context commit：以完整 image context 作为一次调度/slot reservation
  粒度，通过 vLLM `KVCacheManager.allocate_slots()` 申请全部 image token 需要的 slots。
  物理内存仍然是 vLLM 固定大小 KV blocks，不新增按 diffusion 粒度管理的 page manager。
- VAE/artifact decode：不进入 paged KV，不改 block table。

Paged KV 不是 attention 语义边界，但 `torch.sdpa` 不能直接消费 vLLM block table。DiT 对
历史 context 的 attention 直接复用底层 paged attention backend，包装成 read-only、
`causal=False` 的 paged-prefix attention。DiT chunk 内 full attention 使用 dense K/V，并和
paged-prefix attention 通过 LSE merge 合并。

### 2.1.2 KV commit 与 RoPE 约束

AR 和 DiT 的核心区别是 reusable engine KV 的写入时机。

| phase | vLLM paged KV | 进度 |
|---|---|---|
| AR prefill/decode | 每个 scheduled token forward 时立即写入 | `num_computed_tokens` 跟 scheduled tokens 推进 |
| DiT non-final denoise | 不写 vLLM paged KV；chunk 使用 dense scratch K/V，prefix 通过 read-only paged-prefix attention 读取 | 只推进 `dit_step_index` |
| DiT final image-context commit | 一次性写入 generated image context 的 K/V | `num_computed_tokens` 推进到 image tokens 之后 |
| VAE/artifact decode | 不写 vLLM paged KV | 只产生 `materialized_tokens` |

这里的“不写 KV”只指不写入 vLLM engine 的 reusable paged KV。DiT 每个 denoise step 仍会
临时产生 Q/K/V；这些 dense chunk K/V 是 request-local diffusion state，不进入
`SchedulerOutput.num_scheduled_tokens`、不分配 vLLM blocks、不改变 block table。

Denoise 读取文本 context 时，vLLM paged KV 是唯一的 prefix KV source of truth。UAD 不复用
HunyuanImage3 现有 `ImageKVCacheManager` 的 dense prompt-KV cache 路径；UAD 路径需要把
它替换为 read-only paged-prefix attention：DiT Q 是 dense chunk，prefix K/V 按 block
table 从 vLLM paged KV 读取，`causal=False`，不写 KV cache。

RoPE/position 不能用默认 1D append 近似：

- AR 路径的 HunyuanImage3 使用自定义 mRoPE；本地 AR model 已经用
  `get_mrope_input_positions()` 生成 `[T,H,W]` 位置，并用 `HunyuanImage3RotaryEmbedding`
  复现 diffusion 的 interleaved 2D RoPE 规则。
- Diffusion 路径在 `gen_image` 中使用 `build_batch_2d_rope()`、`custom_pos_emb`、
  `position_ids` 和 mixed causal/full `attention_mask`。
- Final image-context commit 必须使用与 denoise/future multiturn 一致的 image grid、
  absolute token layout 和 mRoPE/2D RoPE position。不能只用
  `num_computed_tokens + arange(num_image_tokens)` 当作普通 1D positions。
- 第一版最稳妥的做法：DiT denoise 保持原 diffusion position/mask 体系；final commit 时
  通过 UAD runner 的 HunyuanImage3 helper 生成同一批 image context tokens 及其 mRoPE positions，再写入 vLLM
  paged KV。后续 turn 的 text tokens 接在这条 unified engine token 序列之后。

### 2.2 Step 0：Toy HunyuanImage3 UAD 入口

目标：先把新增模型入口、runner、engine wiring 跑起来。这个 step 不要求真实 DiT，也不
要求和现有 HunyuanImage3 staged serving 对齐。

| 内容 | 说明 |
|---|---|
| engine | 新增 `UADEngine` / `AsyncUADEngine`，接入 request add/step/output lifecycle |
| model | 新增 `HunyuanImage3UADForConditionalGeneration` |
| runner | 新增 `UADRunner`，先只实现 toy AR path；模型私有 phase 规则由 `UADModelStateMachine` 承担，不引入长期独立翻译层 |
| request | 新增 `UADRequestState`，保存 `engine_tokens` 和 `materialized_tokens` |
| output | 文本 token 同时进入 `new_engine_tokens` 和 `new_materialized_tokens` |
| scheduler | 先委托现有 AR scheduler，UAD 只旁路记录 phase |
| 不做 | 不做真实 DiT，不接 paged KV，不做独立翻译层抽象 |

最小 toy 行为：

```text
add_request -> UADEngine.step()
  -> delegated AR scheduler
  -> UAD runner
  -> HunyuanImage3UADForConditionalGeneration toy AR forward
  -> sample one text token
new_engine_tokens = [sampled_token]
new_materialized_tokens = [sampled_token]
num_computed_tokens follows existing AR update timing
```

验证：

- 一个 text-only prompt 能通过 `--uad-engine` 正常输出文本。
- debug log 能确认请求经过 `UADEngine.step()` 和 UAD 模型入口。
- 固定 seed 下输出不需要完全等价原 engine，但 token ledger 不漂。
- `num_computed_tokens`、`new_engine_tokens`、`new_materialized_tokens` 的变化时机能打印
  到 debug log。
- HunyuanImage3 AR-only smoke 能加载模型并跑完 1 个短请求。

退出标准：

- UAD engine / scheduler / runner / request ledger 跑通。
- 后续实现从 runner 继续扩展；当前 toy 单 item 执行层只是 Step 0/2 脚手架，
  Step 3 前必须收进 `UADRunner`。

- 新增 UAD 模型入口被实际调用，而不是只在 engine 外层转发到旧模型。

### 2.3 Step 1：UAD scheduler shadow item

目标：保留原 `SchedulerOutput`，同时生成 UAD runner metadata。runner 仍然只执行
AR toy path。

| 内容 | 说明 |
|---|---|
| 新增 | `UADScheduleItem`、`UADSchedulerOutput` |
| 对应 AR | 复用 `SchedulerOutput.num_scheduled_tokens` 和 request 列表 |
| phase | 根据请求状态标记 `ar_prefill` / `ar_decode` |
| 不做 | 不改 token budget，不调度 DiT，不改 KV allocation / block table |

验证：

- `sum(uad_items.num_scheduled_tokens)` 与 base scheduler 本 tick 的 scheduled tokens 对齐。
- chunked prefill 和 decode 都能产生 UAD item。
- shadow item 能记录 base output 的 `block_ids/new_block_ids/num_computed_tokens` 对应关系。
- 原 AR engine 路径不受影响。

退出标准：

- `--uad-engine` 可以用 UAD metadata 驱动 toy AR forward。

### 2.4 Step 2：Toy phase switch 和统一 token 账本

目标：让 request 能按 HunyuanImage3 的 AR 生成状态机从 AR 切到 DiT phase。DiT
仍然是假执行，但 phase switch、特殊 token、对外 materialize 规则必须先对齐清楚。

HunyuanImage3 现有实现需要对应的规则：

| 来源 | 现有行为 | UAD Step 2 处理 |
|---|---|---|
| `TokenizerWrapper` | 定义 `<boi>`、`<eoi>`、`<img>`、`<cfg>`、`</think>`、`</recaption>`、`</answer>`、`<img_ratio_0>` 和 `special_token_map` | 新增 `HunyuanImage3UADStateConfig`，可从 tokenizer 抽取这些 id；测试用 toy id |
| AR sampler generation mode | `</think> -> <recaption>`，`</recaption> -> <answer><boi><img_size_*>` | config 记录 `stage_transitions` 和 `get_forced_token()`，Step 2 先测试规则，不接真实 sampler |
| AR sampler ratio 约束 | `<img_size_*>` 后只允许 `<img_ratio_*>`；sample 到 ratio 后强制 EOS | ratio token 是 UAD 的 AR->DiT 边界；进入 `dit_step` 后 Step 2 scheduler 不再把 pending image tokens 当 AR 调度 |
| AR sampler comprehension mode | I2T/T2T block `<boi>`、`<eoi>`、`<img_size_*>`、ratio tokens | 记录为后续真实 sampler 对齐项；Step 2 toy 只实现 generation 边界 |
| tokenizer text mask | text section 可带 `ignore=True`，不进入 diffusion text mask | 记录为 Step 4 metadata 对齐项，不在 Step 2 toy 里实现 |
| tokenizer AR KV reuse | `think_recaption_end_pos` 和 `uncond_cfg_start_pos` 标识复用/CFG 边界 | 记录为 Step 4/7 prefix-KV 边界输入，不复用现有 dense `ImageKVCacheManager` |

| 内容 | 说明 |
|---|---|
| 检测 | toy sampler 产出 token 后交给 `HunyuanImage3UADStateMachine`，由模型状态机识别 `<img_ratio_*>` |
| engine tokens | append sampled ratio token，再 append toy `<img>` payload 和可选 `<eoi>`，表示未来要 commit 的 image context |
| materialized tokens | 普通文本 token 对外可见；Hunyuan 结构 token、image/control token、ratio token 不进入 `materialized_tokens` |
| phase | `ar_decode -> dit_step` |
| DiT state | `dit_step_index=0`，`total_dit_steps` 先用很小值，例如 2 |
| KV/page table | toy 阶段只改 ledger，不分配 KV slots，不改 block table |
| scheduler | `dit_step` 请求仍有 pending engine tokens，但 Step 2 先不调度它们，避免误走 AR decode |
| 不做 | 不跑真实 DiT，不做 VAE，不做 cache commit，不接真实 logits processor |

账本语义：

```text
AR sampled token:
  new_engine_tokens += [text_token]
  if token is ordinary text:
      new_materialized_tokens += [text_token]

AR sampled ratio token:
  new_engine_tokens += [ratio_token, image_token_0, ..., image_token_n, optional_eoi]
  new_materialized_tokens unchanged
  phase = dit_step
  dit_step_index = 0
  pending_image_context_commit = true
  num_computed_tokens only advances by the AR tokens scheduled in this tick;
  ratio/image/eoi tokens stay pending until a later cache-commit scheduled item
```

验证：

- 请求能从 AR phase 进入 toy DiT phase。
- ratio token、toy image tokens、`<eoi>` 进入同一条 `engine_tokens` 序列。
- `materialized_tokens` 只包含对外可见普通文本，不包含 Hunyuan 结构 token、ratio token、未 decode image context。
- phase switch 本身不推进 ratio/image/eoi 的 `num_computed_tokens`，不产生 `new_block_ids`。
- scheduler 在 `dit_step` 阶段不会把 pending image context 当 AR item 调度。
- Hunyuan stage-transition helper 和 ratio/EOS 规则有 unit test 覆盖。

退出标准：

- 单请求 phase switch 能跑完，不需要真实图片。
- Step 4 需要把 `TokenizerEncodeOutput` 的 `gen_image_slices`、`gen_image_mask`、
  `gen_timestep_scatter_index`、`think_recaption_end_pos`、`uncond_cfg_start_pos`
  接到 UAD request metadata。

### 2.5 Step 3：Runner-first toy DiT step 调度

目标：先把 Step 0/2 的 toy 单 item 执行职责收进 `UADRunner`，然后让 scheduler 和 runner
支持 “DiT step 是一个可调度 work item”。

| 内容 | 说明 |
|---|---|
| runner cleanup | 删除长期单独翻译层路线；`UADRunner` 直接持有 model 和 model-specific state machine |
| model-specific state machine | HunyuanImage3 在 `HunyuanImage3UADStateMachine` 中定义 `<img_ratio_*>`、stage transition、engine-only token 判断等 phase/output-ledger 规则；runner 不识别这些 token |
| scheduler | DiT step 仍按 token 数消耗 UAD work budget，但 non-final step 不进入 base `SchedulerOutput.num_scheduled_tokens` |
| min/max | toy 阶段令 UAD item 的 `min_tokens == max_tokens == image_query_tokens` |
| runner | 执行 fake DiT step，只更新 `dit_step_index` |
| final step | 标记 fake DiT 完成；Step 2 已经 append 的 image context tokens 等待后续 cache commit |
| KV/page table | non-final DiT step 不分配 KV slots；cache commit item 才需要分配 slots |
| 不做 | 不合并 AR/DiT attention，不跑真实 denoise |

验证：

- 同一个 engine 里，一个 AR request 和一个 toy DiT request 可以交替执行。
- `runner.py` 不再通过独立翻译层执行 item；batch/item 执行入口集中在 `UADRunner.execute_model()`。
- `runner.py` 不直接识别 HunyuanImage3 ratio/control token；AR token 语义委托给
  `UADModelStateMachine`。
- DiT non-final step 只推进 `dit_step_index`。
- DiT final step 不再重复产生 image tokens，也不会假装已经写入 paged KV。
- 后续 cache commit item 必须能通过 scheduler output 携带 block ids。

退出标准：

- UAD scheduler 已经能表达 DiT step，但执行仍是 toy。

### 2.6 Step 4：接入真实 HunyuanImage3 DiT 单请求路径

目标：把 toy DiT step 替换成 HunyuanImage3 的真实模块，但先只支持单请求、固定配置。

| 内容 | 说明 |
|---|---|
| 复用 | `HunyuanImage3Model`、patch/time/final layer、VAE、scheduler timestep |
| init | 根据 AR 产出的 size/ratio 初始化 latents 和 timesteps |
| prefix attention | phase switch/first step 构造 read-only paged-prefix metadata，DiT Q 直接 attend vLLM paged text/prefix K/V |
| step | 一次 UAD DiT item 执行一个独立 denoise timestep；dense scratch K/V 是 request-local state |
| final | final step 默认 commit generated image context，VAE 只产出 artifact |
| KV/page table | non-final step 不碰 vLLM paged KV；final commit 前 scheduler 必须 allocate image token slots |
| position | denoise 保留 diffusion 原生 `position_ids/custom_pos_emb/attention_mask`；final commit 使用 UAD runner helper 生成 image context 的 mRoPE positions |
| 限制 | 单请求、固定 shape、固定 steps、先不做 CFG parallel/SP |

验证：

- 一条 HunyuanImage3 T2I 请求能在 UAD 路径产出图片。
- 不要求 pixel-level 对齐现有 staged pipeline，只要求 seed 固定时行为稳定。
- debug log 能看到每个 DiT step 的 `num_scheduled_tokens`、`dit_step_index`。
- final commit 前后能打印 `block_ids/new_block_ids`、`slot_mapping` 是否生成。

退出标准：

- 新的 UAD 模型入口端到端跑通真实 HunyuanImage3 T2I。

### 2.7 Step 5：同 tick 调度 AR 和 DiT request

目标：scheduler 在一个 tick 内可以同时返回 AR item 和 DiT item。执行可以先按 phase
分组，保证简单可跑。

| 内容 | 说明 |
|---|---|
| scheduler | 保持 vLLM token budget packing，同时增加 UAD work item budget |
| AR item | prefill/decode 语义沿用原 AR scheduler，进入 base `SchedulerOutput` 并分配 KV slots |
| DiT non-final item | 一个 item 对应一个 denoise step，只消耗 UAD work budget，不改 block table |
| DiT final/cache-commit item | 作为 paged-KV scheduled item，需要 `block_ids/new_block_ids/slot_mapping` |
| runner | 先 AR group、DiT group 分别执行 |
| 不做 | 不强行合并 attention/FFN |

验证：

- 构造两个请求：一个 AR decode，一个 DiT denoise，同一 tick 都能被调度。
- budget 不足时 DiT step 可跳过，AR request 仍能继续。
- `num_computed_tokens` 只跟已 commit 的 `engine_tokens` 前缀同步。
- UAD compute-only item 不会出现在 base `SchedulerOutput.num_scheduled_tokens` 中。
- cache-commit item 出现在 base-compatible scheduler output 中，并在 runner 侧更新 block table。

退出标准：

- continuous batching 中 AR 和 DiT request 能共存。

### 2.8 Step 6：统一 HunyuanImage3 UAD layer 调用

目标：开始把 AR token 和 DiT token 放进同一个 UAD 模型入口处理，为 FFN/MoE 合 batch
做准备。

| 内容 | 说明 |
|---|---|
| input | UAD runner 构造一个当前 tick 的 token/hidden buffer |
| AR 部分 | token ids / positions / slot mapping |
| DiT 部分 | latent-derived embeds / timestep embeds / positions |
| paged metadata | AR 和 cache-commit image tokens 复用 vLLM block table / slot mapping |
| attention | 先按语义分开执行，AR 走 causal paged，DiT non-final 走 full/mixed scratch |
| FFN/MoE | 在可以复用同一层权重的位置尝试合并 token batch |

验证：

- AR-only、DiT-only、mixed request 都能跑完。
- mixed 情况下 FFN/MoE trace 的 `local_tokens` 大于 phase-separated 执行。
- 所有写 paged KV 的 tokens 都能在 forward context 中看到合法 `slot_mapping`。
- 输出只做功能检查，不要求覆盖全部 HunyuanImage3 边界。

退出标准：

- UAD 模型入口不再只是转发两个旧路径，已经开始承担统一 layer 调用。

### 2.9 Step 7：Attention core

目标：实现 UAD attention 的第一版正确路径。AR 继续复用 causal paged attention；DiT 直接
实现 read-only paged-prefix attention + dense chunk attention，再做 LSE merge。

设计：

1. AR scheduled tokens 走原 vLLM causal paged self-attention。
2. DiT current chunk 的 Q/K/V 保留 dense scratch tensor。
3. DiT 需要 attend 已 commit prefix 时，调用 read-only paged-prefix attention。该路径复用
   底层 paged attention backend/kernel，输入 DiT dense Q、prefix lengths 和 block table，
   设置 `causal=False`，只读 paged KV，不写 KV cache。
4. DiT chunk 内 attention 使用 `torch.sdpa(Q_chunk, K_chunk, V_chunk)` 或等价 dense full
   attention。
5. final DiT step 可以用 `slot_mapping` 把 image context K/V 写入 paged KV，供下一 turn
   读取；写入前必须已经用正确 mRoPE/2D RoPE positions 生成 K/V，当前 step 不从刚写入的
   paged KV 反读。

DiT attention 合并：

```text
out_ctx,   lse_ctx   = readonly_paged_prefix_attn(Q_chunk, paged_prefix_KV)
out_chunk, lse_chunk = dense_chunk_attn(Q_chunk, K_chunk, V_chunk)
out = lse_merge(out_ctx, lse_ctx, out_chunk, lse_chunk)
```

- `readonly_paged_prefix_attn` 的 Q 是 dense DiT chunk，K/V 来自 vLLM paged KV。
- 它只读 prefix K/V，不写 KV cache，不把 DiT chunk 伪装成 base scheduled tokens。
- 这一步可以复用底层 paged attention backend/kernel，但要构造独立 metadata，并设置
  `causal=False`。因为这里只算 DiT Q 对历史 prefix K/V 的 attention；prefix keys 都在
  chunk 之前，对所有 DiT query 全可见。
- prefix keys 和 chunk keys 是不重叠集合；否则 LSE merge 会双计。
- chunk 内 attention 用 `torch.sdpa` 或等价 dense full attention。
- 这条路径不需要把整个 prefix materialize 成 dense tensor，是 Step 7 的唯一实现路径。

为什么不能直接复用 vLLM base scheduled-token path 给 DiT Q：

- Q 不需要按 KV 的物理 page 顺序排布；paged KV 的物理顺序由 block table 解决。
- Q 必须按 `query_start_loc` 描述的 request 顺序和 request 内 logical token 顺序排布。
- vLLM 现有 `CommonAttentionMetadata` 只有 `query_start_loc`、`seq_lens`、
  `num_computed_tokens`、`block_table`、`slot_mapping`，没有任意 per-query logical position
  mask；base path 默认这些 Q 是 request 当前 scheduled suffix，并且和 scheduled-token
  写 KV 的 `slot_mapping` 绑定。
- 因此不能把任意 DiT dense Q 当作 base `SchedulerOutput.num_scheduled_tokens` 交给
  base self-attention path。
- UAD attention 复用底层 paged attention 计算能力，但包装成 read-only paged-prefix
  attention：输入 DiT Q、prefix lengths 和 block table，`causal=False`，不写 KV cache；
  再与 dense chunk attention 做 LSE merge。

验证：

- 小 shape 下与 dense full mask attention 做数值检查。
- AR-only 请求不进入 DiT attention path。
- DiT chunk 不跨 request attend。
- final commit 后下一 turn 能读到 generated image context。
- DiT attention path 不负责分配 KV slots；KV slots 必须来自 scheduler/runner 的 paged metadata。
- 不依赖现有 vLLM paged FlashAttention 支持混合因果/full mask。
- 对同一 image grid，UAD final commit positions 与 HunyuanImage3 diffusion 2D RoPE /
  AR mRoPE 位置规则一致。

退出标准：

- attention MVP 正确，mixed FFN/MoE path 仍能跑。

### 2.10 Step 8：简单 multiturn smoke

目标：只验证原生 multiturn 的最小账本语义，不追求完整产品边界。

| 事件 | 行为 |
|---|---|
| non-final DiT step | 只更新 latents/image state |
| final DiT step | commit generated image tokens 到 `engine_tokens` |
| VAE epilogue | 只 append image artifact 到 `materialized_tokens` |
| next turn | 新文本继续 append 到同一条 `engine_tokens` 后面 |

验证：

- 第一轮生成图后，第二轮 prompt 能继续进入同一个 request/session ledger。
- `num_computed_tokens` 在 final DiT commit 后推进到 image tokens 之后。
- VAE decode 不改变 `num_computed_tokens`。

退出标准：

- 最小 multiturn smoke 通过。

### 2.11 Step 9：model executor 特性接入

目标：在 toy 和单卡路径跑通后，把 UAD HunyuanImage3 入口接回 vLLM model executor 的
基础能力。这个 step 只做 UAD 作为一个正式 model class 必须具备的能力，不展开 SP/CFG
等额外并行策略。

| 特性 | 第一版要求 |
|---|---|
| model loader | `HunyuanImage3UADForConditionalGeneration` 能通过显式 UAD 开关被加载 |
| weight loading | 复用现有 HunyuanImage3 AR 权重名和必要的 diffusion 权重 mapping |
| TP | AR dense/LM head 路径沿用现有 TP；DiT 先支持现有 HunyuanImage3 DiT TP 配置 |
| PP | 不作为第一版目标；如果现有 AR PP 路径天然可用则保留，不为 UAD 新增 PP 逻辑 |
| quant | 先支持现有 HunyuanImage3 已验证 quant 配置，不新增 UAD 专属 quant kernel |
| model registry | 先走 `--uad-engine` 显式选择；稳定后再决定是否注册成独立 architecture |
| profiling | trace 能区分 AR layer、DiT layer 和 UAD mixed layer |

验证：

- `TP=1` 跑通 AR-only、T2I single request、最小 multiturn smoke。
- `TP=2/4` 跑通 AR-only 和 T2I single request。
- quant 配置下至少完成单请求 smoke。
- UAD 入口实际经过 `model_executor/models/hunyuan_image3/hunyuan_image3_uad.py`。

退出标准：

- UAD HunyuanImage3 不再只是单卡 research stub，已经能复用基础 TP/quant/model loading
  能力。

### 2.12 Step 10：性能回归和 motivation 对照

复用第 1 节 trace 工具，对比三条路径：

1. 现有 staged HunyuanImage3。
2. `UADEngine` step-level 但 phase-separated runner。
3. `UADEngine` mixed FFN/MoE runner。

核心对比图：

- stage idle ratio before/after。
- FFN/MoE token batch time series before/after。
- latency p50/p99 vs request rate。
- throughput vs request rate。
- GPU SM utilization vs request rate。
- FFN/MoE microbench saturation threshold overlay。

验收标准：

- 正确性：AR-only、T2I single request、最小 multiturn smoke 都能正常跑完。
- 性能：mixed FFN/MoE runner 的 FFN token batch 分布明显右移。
- 稳定性：在至少 3 个 request rate 下长跑无 deadlock、无 request ledger drift。

## 3. 初始不做的内容

第一版先不做：

- CFG parallel 在 UAD scheduler 中的 branch 表达。
- SP / Ring / Ulysses 在 UAD attention patch 下的完整设计。
- EP 和 SP+EP 的 UAD mixed runner 正式支持。
- 强制 AR + DiT attention 单 kernel。
- 跨节点 KV transfer 重设计。
- VAE decode 进入 scheduler token budget。

这些保留为后续 RFC。第一版只证明：同一个 engine 内，AR 和 DiT request 能共存，
并且 FFN/MoE 可以获得比 staged serving 更大的有效 batch。
