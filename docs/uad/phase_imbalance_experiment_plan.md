# HunyuanImage3 Phase Imbalance 实验计划

目标：验证固定 AR/DiT 分离部署在在线流量下是否会因为 phase 比例变化导致资源不均衡：
一个 stage 排队，另一个 stage 空闲或低负载。实验只讨论现有 vLLM-Omni HunyuanImage3
分离 serving，不引入 UAD 执行器。

## 0. Gate：先确认 phase 内 continuous batching

公平对比需要先确认每个 phase 内部已经能 batch，否则测到的空闲可能只是单 phase batching
缺失，而不是 AR/DiT 固定切分的问题。

检查命令：

```bash
python docs/uad/script/check_hunyuan_phase_batching.py \
  --output-json artifacts/uad_phase_imbalance/preflight/phase_batching.json
```

当前最新 upstream 合入后的检查结果：

- 代码基线：`origin/main` = `c99df1eb`，实验分支 merge commit = `0cc53678`。
- AR 模型代码已经支持 batched sampler：`sample()` 对 `logits.shape[0]` 逐 request
  处理，旧的单 batch 断言已不存在。对应 upstream 修复是
  `f7161b07 [Bugfix]Allow HunyuanImage3 AR sampler batching (#3590)`。
- 默认 `vllm_omni/deploy/hunyuan_image3.yaml` 仍设置 stage 0 `max_num_seqs: 1`，
  实验需要 overlay 成 `max_num_seqs > 1` 才会真正打开 AR phase 内 continuous batching。
- HunyuanImage3 DiT pipeline 当前没有 `supports_step_execution=True`，也没有
  `prepare_encode / denoise_step / step_scheduler / post_decode` stepwise 接口。
- `DiffusionEngine` 对非 stepwise diffusion 会把 `max_num_seqs > 1` 强制降回 1。
  因此 HunyuanImage3 DiT 当前不能做多个在线请求的 denoise-step continuous batching。
- 默认 AR->DiT edge `max_inflight: 1`，实验 overlay 需要改成大于 1，否则 stage 边界也会
  人为限制并发。

结论：当前代码可以跑“现有分离部署的 phase imbalance”实验，但不能声称已经开启了
HunyuanImage3 的 DiT phase 内 continuous batching。若要做严格公平实验，需要先实现或接入
HunyuanImage3 DiT stepwise execution，然后再运行完整 rate sweep。

## 1. 要验证的假设

H1：固定 `G_ar + G_dit` 分离部署下，在线 workload 的 AR/DiT work ratio 不是常数，会出现
`AR queue 堵且 DiT GPU 空` 或 `DiT queue 堵且 AR GPU 空`。

H2：这种不均衡在 SLO goodput 视角比裸 throughput 更明显。只看 throughput 时可以持续送请求，
但请求可能已经排队超时；SLO goodput 会因为 bottleneck phase 排队而下降。

H3：如果未来 UAD 让每个 DP group 都能执行 AR 和 DiT，且单 phase 执行效率接近分离部署，
那么对 phase-ratio 波动 workload，UAD 的 capacity pooling 应该能减少 stranded capacity。

## 2. 数据集

主数据集：

- DiffusionDB metadata-only prompt：真实 text-to-image prompt 分布，过滤 NSFW，
  按 Hunyuan tokenizer 长度分桶。

补充数据集：

- PartiPrompts/P2：固定的通用 text-to-image prompt 集，便于复现实验。
- GenEval prompts：短 prompt / compositional prompt，构造 DiT-heavy workload。
- DPG-Bench 或自生成 dense prompt：长细节 prompt，构造 AR-heavy workload。

所有数据集统一生成 JSONL：

```json
{"request_id":"...", "prompt":"...", "width":1024, "height":1024, "steps":50, "guidance_scale":0.0, "seed":42}
```

## 3. Workload

使用 open-loop Poisson arrival，避免 closed-loop 客户端把 server 排队隐藏掉。

三类 workload：

- DiT-heavy：短 prompt，`1024x1024`，`steps=50`。预期 DiT 堵，AR 空。
- AR-heavy：长 prompt，`512x512` 或较少 steps。预期 AR 相对更重。
- Bursty-mix：每 5 分钟切换比例，例如 `80% DiT-heavy -> 80% AR-heavy -> 50/50`。

request rate sweep：

```text
0.25x, 0.5x, 0.75x, 0.9x, 1.0x, 1.1x, 1.25x
```

其中 `1.0x` 先用短 warmup sweep 找到接近 SLO knee 的速率。

## 4. 部署配置

Baseline：现有 HunyuanImage3 分离部署，2 卡 AR + 2 卡 DiT：

```bash
vllm-omni serve tencent/HunyuanImage-3.0-Instruct \
  --deploy-config vllm_omni/deploy/hunyuan_image3.yaml \
  --host 0.0.0.0 --port 8091
```

Continuous-batching overlay 需要：

- stage 0：`max_num_seqs > 1`，`max_num_batched_tokens` 足够大。
- stage 1：必须先支持 HunyuanImage3 DiT `step_execution: true`，否则配置会被降回单请求。
- edge 0->1：`max_inflight > 1`。

当前 Gate 0 结论是 stage 1 未满足，所以完整公平实验先标记为 blocked。

## 5. 指标

硬件指标：

- DCGM `SM_ACTIVE`
- DCGM `PIPE_TENSOR_ACTIVE`
- DCGM `DRAM_ACTIVE`
- power / memory used

stage 指标：

- `stage_busy_ratio = sum(forward_time_ms) / wall_time_ms`
- AR / DiT queue length
- running request count
- AR finish -> DiT start handoff wait
- per-forward FFN/MoE token count

核心不均衡指标：

```text
AR_stranded(t)  = AR_idle(t)  and DiT_queue_len(t) > 0
DiT_stranded(t) = DiT_idle(t) and AR_queue_len(t) > 0

stranded_capacity_ratio =
  time(AR_stranded or DiT_stranded) / total_measurement_time
```

SLO goodput：

```text
goodput = completed_requests_within_E2E_SLO / measurement_seconds
```

建议 SLO 先设为 baseline p90 latency 的 1.2x / 1.5x 两档。

## 6. 执行顺序

1. Gate 0：运行 `check_hunyuan_phase_batching.py`。
2. 如果 Gate 0 未通过：
   - 只运行现有 baseline smoke，记录当前分离部署能力和缺口。
   - 不运行“公平 continuous batching”主实验。
3. 如果 Gate 0 通过：
   - 启动 2+2 HunyuanImage3 online serving。
   - 运行 dataset smoke：每类 workload 8 个请求。
   - 运行 rate sweep：每个点 warmup 3 分钟，measurement 15 分钟，重复 3 次。
   - 生成 utilization / queue / stranded capacity / goodput HTML 报告。

## 7. 当前执行记录

已执行 Gate 0，结论是 blocked：AR 代码可 batch，但默认配置未打开；DiT 不支持
HunyuanImage3 stepwise batching，非 stepwise diffusion 会强制单请求执行。

下一步需要在两条路线中选择：

- 路线 A：先实现 HunyuanImage3 DiT stepwise execution，再做严格公平实验。
- 路线 B：先测当前生产形态的分离部署 imbalance，并在报告中明确 DiT phase 内未 continuous batch。
