# HunyuanImage3 Phase Imbalance 实验计划

目标：验证固定 AR/DiT 分离部署在在线流量下是否会因为 phase 比例变化导致资源不均衡：
一个 stage 排队，另一个 stage 空闲或低负载。实验只讨论现有 vLLM-Omni HunyuanImage3
分离 serving，不引入 UAD 执行器。

当前先做 Version A：测真实现状。HunyuanImage3 DiT 仍是 request-mode 单请求执行；
其他 DiT 请求会在 diffusion scheduler waiting queue 里排队。Version A 不声称 DiT
phase 内已经 continuous batching，只回答“当前分离部署在在线流量下是否出现阶段负载不均衡”。

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

Baseline：现有 HunyuanImage3 分离部署，2 卡 AR + 2 卡 DiT。先生成一份 Version A
实验 deploy YAML：打开 AR stage 和 AR->DiT edge 并发，但保留 DiT 单请求。

```bash
python docs/uad/script/make_hunyuan_phase_deploy_config.py \
  --base vllm_omni/deploy/hunyuan_image3.yaml \
  --output artifacts/uad_phase_imbalance/config/hunyuan_image3_phase_a.yaml \
  --ar-max-num-seqs 8 \
  --ar-max-num-batched-tokens 32768 \
  --dit-max-num-seqs 1 \
  --edge-max-inflight 64
```

启动 serving：

```bash
vllm-omni serve tencent/HunyuanImage-3.0-Instruct \
  --deploy-config artifacts/uad_phase_imbalance/config/hunyuan_image3_phase_a.yaml \
  --host 0.0.0.0 --port 8091
```

完整 continuous-batching overlay 需要：

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

## 6. Version A 执行顺序

### 6.1 Gate 0

```bash
python docs/uad/script/check_hunyuan_phase_batching.py \
  --output-json artifacts/uad_phase_imbalance/preflight/phase_batching.json
```

### 6.2 生成 workload

```bash
python docs/uad/script/make_hunyuan_phase_workload.py \
  --profile dit_heavy \
  --num-requests 256 \
  --output artifacts/uad_phase_imbalance/workloads/dit_heavy.jsonl

python docs/uad/script/make_hunyuan_phase_workload.py \
  --profile ar_heavy \
  --num-requests 256 \
  --output artifacts/uad_phase_imbalance/workloads/ar_heavy.jsonl

python docs/uad/script/make_hunyuan_phase_workload.py \
  --profile bursty_mix \
  --num-requests 512 \
  --output artifacts/uad_phase_imbalance/workloads/bursty_mix.jsonl
```

脚本 dry run：

```bash
python docs/uad/script/run_hunyuan_phase_load.py \
  --workload artifacts/uad_phase_imbalance/workloads/dit_heavy.jsonl \
  --output-jsonl artifacts/uad_phase_imbalance/dryrun/dit_heavy.jsonl \
  --rate 0.1 \
  --max-requests 8 \
  --dry-run
```

### 6.3 Rate sweep

每个 workload 从低到高扫 rate。先用短 smoke 找到 knee，再做长 measurement。示例：

```bash
python docs/uad/script/run_hunyuan_phase_load.py \
  --workload artifacts/uad_phase_imbalance/workloads/dit_heavy.jsonl \
  --output-jsonl artifacts/uad_phase_imbalance/results/dit_heavy_r0.05.jsonl \
  --base-url http://127.0.0.1:8091 \
  --rate 0.05 \
  --duration-s 600 \
  --timeout-s 1800 \
  --nvidia-smi-jsonl artifacts/uad_phase_imbalance/results/dit_heavy_r0.05_gpu.jsonl \
  --metrics-jsonl artifacts/uad_phase_imbalance/results/dit_heavy_r0.05_metrics.jsonl
```

推荐第一轮 rates：

```text
0.02, 0.05, 0.08, 0.10, 0.12, 0.15 req/s
```

如果 serving 已经明显排队，再围绕 knee 加密；如果所有点都空闲，再继续升高。

### 6.4 生成报告

```bash
python docs/uad/script/build_phase_imbalance_report.py \
  --result-jsonl artifacts/uad_phase_imbalance/results/dit_heavy_r0.05.jsonl \
  --gpu-jsonl artifacts/uad_phase_imbalance/results/dit_heavy_r0.05_gpu.jsonl \
  --output-html artifacts/uad_phase_imbalance/report/dit_heavy_r0.05.html \
  --summary-json artifacts/uad_phase_imbalance/report/dit_heavy_r0.05_summary.json \
  --ar-gpus 0,1 \
  --dit-gpus 2,3 \
  --slo-s 120 \
  --slo-s 180 \
  --slo-s 300
```

多个 rate 可以把 `--result-jsonl` 和 `--gpu-jsonl` 重复传入同一个报告命令。

## 7. 当前执行记录

已执行 Gate 0，结论是 blocked for full continuous batching：AR 代码可 batch，但默认配置未打开；
DiT 不支持 HunyuanImage3 stepwise batching，非 stepwise diffusion 会强制单请求执行。

当前执行 Version A：使用现有分离部署能力做 baseline imbalance 实验，并在报告里明确 DiT
phase 内未 continuous batch。后续 Version B 再实现 HunyuanImage3 DiT stepwise execution，
用于严格比较“分离但 phase 内可连续 batch”与 UAD 的差异。
