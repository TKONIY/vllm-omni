# DreamZero End-to-End Last Steps

本文记录的是：**DreamZero 在 `vllm-omni-wm` 里的正式 OpenPI 服务当前已经做到什么程度，以及还剩哪些真正没做完的事。**

## 0. 当前结论

- **DreamZero-DROID 已经在正式 `vllm serve --omni` 路径上跑通。**
- **正式 `/v1/realtime/robot/openpi` WebSocket 链路已通过 source-vs-vLLM parity 测试。**
- **同一份 DreamZero client 逻辑现在可以同时连原版 DreamZero server 和 vLLM server；客户端唯一差异只是 websocket path。**
- 当前严格对齐口径已经固定为：
  - DreamZero upstream eager
  - 不加 `torch.compile`
  - 不启用 DiT cache / skip schedule
  - `TP=1`

### 0.1 已经正式验证通过的链路

- `vllm serve --omni`
- `AsyncOmni`
- `StageDiffusionProc`
- `DiffusersPipelineLoader`
- `DreamZeroPipeline.forward()`
- `/v1/realtime/robot/openpi`
- `infer -> infer -> reset -> infer`

### 0.2 当前已经通过的正式测试

- `tests/dreamzero/test_client_ar_path_parity.py`
  - 验证同一个 `tests/dreamzero/test_client_AR.py` 在连 upstream `/` 和 vLLM `/v1/realtime/robot/openpi` 时，客户端侧 observation / infer / reset 逻辑保持一致
- `tests/dreamzero/test_openpi_e2e_source_parity.py`
  - 验证 upstream DreamZero server vs `vllm serve --omni`
  - 当前通过口径：
    - DROID
    - `TP=1`
    - eager / no-compile / no-DiT-cache
    - `rtol=0.0, atol=0.0`

### 0.3 本轮额外确认并修掉的真实问题

- `DreamZeroPipeline` 构造 `CausalWanModel` 时，曾错误把 `action_head_cfg.config.hidden_size=64` 传进 DiT
- 这会把本地 `action_decoder.layer1.W` 形状缩成 `(1, 5120, 64)`，从而在正式 stage 启动时加载 DreamZero root 权重失败
- 现已修正为：
  - DiT 严格按 `diffusion_model_cfg` 构造
  - 不再把 action-head 级 `hidden_size=64` 误传给 `CausalWanModel`

---

## 1. 已完成项

### 1.1 P0 基础接线

以下原先的 P0 阻塞项现在都已经完成：

- `registry.py` 已注册 `DreamZeroPipeline`
- `stage_diffusion_proc.py` 已能把 DreamZero root 正确识别为 `DreamZeroPipeline`
- `od_config.model_path` 已统一收敛为 `od_config.model`
- `DreamZeroPipeline.__init__()` 已直接读取 root `config.json`
- `experiment_cfg/metadata.json` 已接入 action/state normalization
- `weights_sources` 已指向 DreamZero root
- `load_weights()` 已实现 4 类 root 权重 remap：
  - `action_head.model.*`
  - `action_head.text_encoder.*`
  - `action_head.image_encoder.*`
  - `action_head.vae.*`

### 1.2 组件初始化策略

当前组件初始化策略已经稳定下来：

- `tokenizer`
  - 默认直接走 `google/umt5-xxl`
- `text_encoder`
  - 本地构造 `UMT5EncoderModel(config)`
  - 最终权重来自 DreamZero root `action_head.text_encoder.*`
- `image_encoder`
  - 正式实现使用仓内 `DreamZeroImageEncoder`
  - 最终权重来自 DreamZero root `action_head.image_encoder.*`
- `vae`
  - 可显式给 source，也可直接构造 `DistributedAutoencoderKLWan()`
  - 最终权重来自 DreamZero root `action_head.vae.*`
- `transformer`
  - 本地构造 `CausalWanModel(config)`
  - 最终权重来自 DreamZero root `action_head.model.*`

这意味着：

- **现在已经不需要 prepared bundle 才能启动**
- **用户只给官方 HF DreamZero root repo id 也能走正式服务路径**

### 1.3 服务链路与客户端兼容性

以下也已完成：

- 正式 `/v1/realtime/robot/openpi` 路由已注册
- DreamZero warmup dummy request 已处理
- action 输出已走正式 diffusion output 契约，不再依赖 fallback 零值路径
- 同一个 DreamZero client 脚本可以：
  - 连原版 DreamZero server：默认 `path=""`
  - 连 vLLM server：`--path /v1/realtime/robot/openpi`

---

## 2. 现在真正还没做完的事

下面这些才是当前还值得保留的未完成项。

### 2.1 `TP=2` 严格精度对齐

现状：

- `TP=1, CF_P=1`：严格对齐
- `TP=1, CF_P=2`：严格对齐
- `TP=2, CF_P=1/2`：可运行，但不严格对齐

已知主因：

- vLLM 原生 `RowParallelLinear` 在 `bf16 + TP>1` 下有数值漂移
- DreamZero pipeline 中首个稳定失败点是：
  - `state.positive.kv[1]`
  - `max_diff = 1.562e-02`

结论：

- **如果目标只是“功能支持 TP=2”**，当前已经达到
- **如果目标是“TP=2 也和 upstream eager 严格一致”**，这项还没做完

### 2.2 正式服务 e2e 覆盖面还不完整

当前正式 e2e 已完成的是：

- DROID
- `TP=1`
- eager / no-compile / no-DiT-cache

还没补成正式自动化覆盖的包括：

- `CF_P=2` 的正式 OpenPI e2e
- AgiBot 的正式 smoke / parity
- `TP=2` 的正式服务 smoke（即使不追 strict parity，也可以补“可运行”回归）

### 2.3 性能路径暂未接回

还没做：

- DiT cache / static skip mask
- dynamic cache schedule / cosine skip
- async predicted-frame feedback 路径

这些都不影响当前 eager parity 基线，但属于后续性能项。

### 2.4 并行框架复用仍未完成

还没做：

- sequence parallel / Ulysses / ring parallel
- 像 WAN2.2 / Bagel 那样的并行切分复用

当前 DreamZero 端口还是以现有 TP / CFG 并行为主。

### 2.5 DreamZero compiled 路径不在当前对齐目标内

当前只对齐：

- DreamZero eager

还没做：

- DreamZero upstream compiled scheduler / compiled full path 的数值对齐

这块差异已经定位并记录，但没有实现“compiled parity”。

---

## 3. 如果接下来继续做，推荐顺序

### 3.1 第一优先级

- 补 `CF_P=2` 的正式 OpenPI e2e
- 补 AgiBot 的正式 smoke

### 3.2 第二优先级

- 决定 `TP=2` 的目标口径：
  - 如果接受“可运行但不 strict parity”，那文档即可
  - 如果要求 strict parity，就要继续追 `RowParallelLinear bf16` 路径

### 3.3 第三优先级

- 接回 DiT cache / skip schedule
- 接 sequence / ring parallel
- 如果有明确需求，再追 DreamZero compiled parity

---

## 4. 一句话总结

**当前真正已经完成的是：DreamZero-DROID 在正式 vLLM OpenPI 服务路径上、用官方 HF root、在 eager / TP=1 基线下，已经和 upstream DreamZero server 端到端严格对齐。**

**当前真正还没做完的是：`TP=2` strict parity、`CF_P=2`/AgiBot 正式 e2e 覆盖、以及性能路径与更高级并行能力。**
