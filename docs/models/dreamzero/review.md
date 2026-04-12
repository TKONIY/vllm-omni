# DreamZero Review

## 结论

- `causal_wan_model.py` 的模型级对齐结论保持成立：旧 review 里关于 self-attention / cross-attention / KV cache / TP=1 / TP=2 / tiny full model 的结论没有被这次改动推翻。
- `pipeline_dreamzero.py` / `state_dreamzero.py` / `transform/*` 的基础接线问题已经大幅收敛：输入契约、UMT5/VAE 接法、negative prompt、deterministic seed、serving reset、AR 帧累积、state/embodiment_id 形状、以及 action 后处理都已补齐。
- 当前工作树下，pipeline 级 parity 已经可以给出明确结论：
  - `TP=1, CF_P=1/2`：与 DreamZero eager 参考实现严格对齐；
  - `TP=2, CF_P=1/2`：首个失败点稳定出现在 `state.positive.kv[1]`，`max_diff=1.562e-02`。
- 当前对齐口径已收敛为：**DreamZero eager + 不加 `torch.compile` + 不启用 DiT cache/skip schedule**。仓内 `scheduling_flow_unipc_multistep.py` 已回退到原仓版本，且单独测试确认它与 DreamZero eager scheduler 在 step0/step1 上完全一致；DreamZero upstream 的 compiled/eager `bf16` 数值差异单独记录在 `todo.md`。
- 正式服务链路现在也已经有结论，而不再只是 pipeline 级结论：
  - `tests/dreamzero/test_client_ar_path_parity.py` 已确认同一份 DreamZero client 脚本在 upstream `/` 与 vLLM `/v1/realtime/robot/openpi` 两种 path 下保持相同客户端逻辑；
  - `tests/dreamzero/test_openpi_e2e_source_parity.py` 已确认 upstream DreamZero server vs `vllm serve --omni` 在 DROID、`TP=1`、eager/no-compile/no-DiT-cache 基线下端到端严格对齐。

## 本轮新增确认并已修复

### 正式 stage 启动时的 `hidden_size` 误接线

- `DreamZeroPipeline.__init__()` 曾把 `action_head_cfg.config.hidden_size=64` 误传给 `CausalWanModel`
- 这会错误缩小 DiT 内部 action/state MLP 维度，并在正式 stage 加载 root checkpoint 时触发形状不匹配
- 当前已修复为：
  - DiT 只按 `diffusion_model_cfg` 构造
  - 不再把 action-head 级 `hidden_size` 传入 `CausalWanModel`
- 该修复已经被正式服务 e2e 覆盖到：修复后 `vllm serve --omni` 的 DreamZero stage 能正常初始化并通过 source parity

## 本次确认已解决，并已从问题列表移除

### 输入/组件接线

- `pipeline_dreamzero.py` 已改为消费 transform 输出的 `prompt / images / state / embodiment_name`，不再依赖旧的 `text / attention_mask / embodiment_id` 契约。
- `pipeline_dreamzero.py` 已改用仓内可复用的 `transformers.UMT5EncoderModel` 与 `DistributedAutoencoderKLWan`；旧 review 里“导入路径错误、运行时无法构造”的问题已解决。
- `_encode_text()` 已按仓内 UMT5 的真实接口读取 `.last_hidden_state`，旧 review 里这条问题已解决。

### 与 DreamZero 入口链路对应的改动

- `transform/base.py`、`transform/droid.py`、`transform/roboarena.py` 现在已经承接了 DreamZero `dreamzero_cotrain.py` 里真正属于 transform 层的逻辑：
  - 多视角拼接；
  - embodiment-aware language template；
  - raw state 提取；
  - 输出 action dim 裁剪。
- `pipeline_dreamzero.py` 已补上固定 negative prompt，对应 `dreamzero_cotrain.py` `apply_single()` 的 uncond 文本准备。
- `pipeline_dreamzero.py` 已补上固定 `seed=1140` 的 deterministic noise，对应 `wan_flow_matching_action_tf.py` 的固定噪声路径。
- `openpi_serving.py -> pipeline_dreamzero.py` 的 `extra_args["reset"]` 已打通，session 切换不再完全绕过 pipeline state。
- `pipeline_dreamzero.py` / scheduler 对外链路里手传 `step_index` 的改动已经删干净；当前只保留 scheduler 内部计数器，这一层是 diffusers 风格内部状态，不再暴露到 DreamZero pipeline API。

### AR / action 条件分支修复

- `state_dreamzero.py` 已从旧的“每路相机 buffer”适配成当前 serving 链路真正需要的“stitched frame buffer”；`pipeline.forward()` 也已经实际调用 `state.accumulate_frames()`，AR 帧累积不再悬空。
- `state_features` 已 pad 成 `(B, 1, max_state_dim)`，`embodiment_id` 已变成 `(B,)` tensor，符合 `CategorySpecificMLP` / `CategorySpecificLinear` 的输入契约。
- action 后处理现在与 `sim_policy.py` 的语义一致：
  - 先做 q01/q99 反归一化；
  - 只给前 `relative_action_dim=7` 个 joint 维度加回 last state；
  - 不再错误地把 gripper 也当作 relative action；
  - 最终输出 `(horizon, ACTION_DIM)`。

## 当前仍未对齐的点

### [已澄清] `image_encoder` 的真正结论：当前应走本地 `DreamZeroImageEncoder`

- 新增 `tests/dreamzero/test_hf_image_encoder_real_weight_parity.py` 后，real-weight / real-input 的严格结论已经明确：
  - upstream `WanImageEncoder.encode_image()`
  - 当前本地端口 `DreamZeroImageEncoder.encode_image()`
  - 手工构造并从 DreamZero root 权重 remap 的 `CLIPVisionModel(...).hidden_states[-2]`
- 其中稳定成立的结论是：
  - 本地 `DreamZeroImageEncoder` 在真实 checkpoint 权重 + 真实服务输入路径下与 upstream `max_diff = 0.0`；
  - `CLIPImageProcessor` 不是 DreamZero 源码等价预处理；
  - HF `CLIPVisionModel` 路线在当前真实 `bf16` 服务输入口径下仍会漂移，因此不应作为当前正式实现。
- 因此，当前 pipeline 的正确实现是：
  - 直接复用仓内 `DreamZeroImageEncoder`；
  - 权重按源码命名直接映射：`action_head.image_encoder.* -> image_encoder.*`；
  - 不再走 HF `CLIPVisionModel` 作为生产路径。

### [高] `TP>1` 的 pipeline parity 仍受 `RowParallelLinear` `bf16` 漂移影响

- 最新 pipeline parity 实测：
  - `TP=1, CF_P=1`：通过；
  - `TP=1, CF_P=2`：通过；
  - `TP=2, CF_P=1`：失败于 `state.positive.kv[1]`，`max_diff=1.562e-02`；
  - `TP=2, CF_P=2`：失败于 `state.positive.kv[1]`，`max_diff=1.562e-02`。
- 这个结果与现有 `RowParallelLinear` 专项测试一致：`bf16 + TP>1` 会先在层级测试出现明显偏差，再在 DreamZero pipeline 中首先体现为 KV cache 漂移。
- 因此，当前阻塞“整条链路 TP=2 严格对齐”的主因已经收敛到 TP 线性层数值路径，而不是 pipeline / state / CFG / image encoder / scheduler 接线错误。

### [中] 正式服务 e2e 覆盖面仍未补全

- 现在已经完成的正式服务 parity 覆盖是：
  - DROID
  - `TP=1`
  - eager / no-compile / no-DiT-cache
- 还没有补成正式自动化回归的包括：
  - `CF_P=2` 的 OpenPI e2e
  - AgiBot 的正式 smoke / parity
  - `TP=2` 的正式服务 smoke（即便不追 strict parity，也可以补“可运行”回归）

### [已修复] ~~CFG parallel prefill 不是上游语义~~ → 已对齐

- `_prefill_kv_cache()` 现在走 `predict_noise_maybe_with_cfg()` — 与 denoise loop 完全相同的路径，遵循 `cfg_parallel.md` 的设计规则。
- mixin 自动处理 rank 分发：`cfg_world_size > 1` 时 rank 0 只 prefill cond，rank 1 只 prefill uncond。
- KV cache 更新作为 `predict_noise()` 的 side effect 发生（`update_kv_cache=True`），不再手动判断 rank。
- 不再有任何手写的 `get_classifier_free_guidance_rank()` 检查在 prefill 里。

### 模型构造一致性修复

- `causal_wan_model.py` 现在只在 `model_type == "i2v"` 时创建 `img_emb`，已与 DreamZero `wan_video_dit_action_casual_chunk.py:1380-1381` 对齐。
- `causal_wan_model.py` 已补回上游 `init_weights()` 裸模型初始化路径，并将同样的 Xavier/zero 规则适配到 vLLM `ColumnParallelLinear` / `RowParallelLinear`。
- `DistributedRMSNorm` 已改回与 upstream `WanRMSNorm` 同形的 `x * rsqrt(mean_sq + eps)` 数值路径，不再使用 `sqrt + divide` 变形；这修掉了 `TP=1`、`no-skip` 基线下首个 `block1.self_attn.norm_k` 漂移点，并恢复了 `tests/dreamzero/test_pipeline_dreamzero_parity.py` 的 `TP=1, CF_P=1/2` 严格对齐。

### cross-attn cache 语义澄清

- `WanT2VCrossAttention` / `WanI2VCrossAttention` 模块本身支持 `crossattn_cache`，单卡与 TP=2 都已经补测了 cache 的 fill/reuse 语义。
- 但 DreamZero `causal_wan_model.py` 这条 full-model 推理链路不会把 `crossattn_cache` 继续传进 block / cross-attention；上游 `wan_video_dit_action_casual_chunk.py` 也是同样语义。
- 因此，full-model 测试里正确的断言是 `crossattn_cache` 保持未初始化，而不是“prefill 后被填充、下一步被复用”。

### [已定位] DreamZero scheduler 的 compile/eager 差异已独立出来记录

- 已完成的缩圈实验表明：
  - 当前仓内 scheduler（恢复到原仓版本后）与 DreamZero eager scheduler：`step0 = 0.0`，`step1 = 0.0`；
  - 直接对 DreamZero 上游做 A/B：`torch.compile` 与 identity-compile 在同一组 CUDA `bf16` 输入上，`step0_max_diff=1.562500e-02`，`step1_max_diff=1.562500e-02`；
  - 再缩到最小表达式：仅 `a / b * x`（`a,b` 为 `float32` 标量，`x` 为 CUDA `bf16` 张量）时，compiled / eager 已经出现 `1.562500e-02` 差异；换成 `float32` 输入后差异降到 `0`。
- 结论：
  - 这里没有发现新的 scheduler 逻辑错误；compiled/eager 的首个分歧点在 predictor 式子 `x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0` 的 `bf16` GPU 算子舍入路径；
  - 这次端口目标只要求对齐 DreamZero eager 路径，因此 `scheduling_flow_unipc_multistep.py` 不需要保留额外的 compiled 特化改动；
  - 如果未来要进一步追 DreamZero compiled 路径，需要单独处理这部分数值差异。

### [中] 旧 review 里的并行框架复用问题仍然成立

- DreamZero 端口依然没有像 WAN2.2 / Bagel 那样接入现成的 sequence parallel / ring parallel 规划。
- `causal_wan_model.py` 里 `skip_sequence_parallel=True` 与缺少 `_sp_plan` 的问题仍在。
- 这不直接否定当前模型级精度结论，但从“并行框架复用程度”看，旧 review 这部分结论仍成立。

## 本次复审后，旧问题的处理结论

- 可以明确从旧问题列表删除的项：
  - `pipeline` / `transform` 输入 key 契约不一致；
  - `UMT5` / `VAE` 导入路径错误；
  - `_encode_text()` API 用错；
  - fixed negative prompt 缺失；
  - deterministic seed 缺失；
  - serving reset 未传到 pipeline；
  - AR 帧累积未接通；
  - `state` / `embodiment_id` 形状不对；
  - action 后处理仍是 raw padded action。
- 当前真正还应该保留在 review 里的问题，应该只剩：
  - `TP>1` 下 `RowParallelLinear` 引入的 pipeline parity 漂移；
  - `CF_P=2` / AgiBot / `TP=2` 的正式服务级自动化覆盖仍未补齐；
  - sequence/ring parallel 框架未复用。
- 已从问题列表移除（本轮修复）：
  - CFG parallel prefill 语义 → 已改为走 `predict_noise_maybe_with_cfg()`，遵循 `cfg_parallel.md` 设计规则。
  - `image_encoder` 路线 → 已切回仓内 `DreamZeroImageEncoder`，并补齐 `action_head.image_encoder.* -> image_encoder.*` 直映射。
  - CFG 数学里的 `cfg_normalize=False` → 已显式补到 prefill / denoise 调用点，对齐 DreamZero 直接 CFG 公式。
  - `crossattn_cache` 测试口径错误 → 已拆清“模块级支持 cache”与“full-model 链路不上 cache”两层语义，并补上对应测试。
  - pipeline 级 TP / CF_P / upstream parity 测试缺失 → 已补上 `tests/dreamzero/test_pipeline_dreamzero_parity.py`，当前口径为 DreamZero eager reference，并已拿到明确通过/失败矩阵。
  - scheduler 外部 `step_index` 传递 → 已删除，对 parity 无影响。
  - DiT cache / skip schedule 代码 → 已从 `pipeline_dreamzero.py` 删除；当前基线明确只对齐 no-compile / no-skip 的 eager 路径。
