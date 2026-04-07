# DreamZero Implementation TODO

## 当前状态

- `action_encoder.py`：已完成，对齐 DreamZero，GPU 精度测试通过。
- `causal_wan_model.py`：inference-only 端口已完成，热点路径已经切到 vllm-omni 基础设施；单卡与 TP=2 的 `fp32`/单层级测试通过，但保留原生 `bf16 x bf16` 的 `RowParallelLinear` 后，`.venv` 下 pipeline parity 在 `TP>1` 仍会出现数值漂移，详见下方 TODO。
- `pipeline_dreamzero.py` / `state_dreamzero.py` / `transform/*`：主链路已接通；当前 `tests/dreamzero/test_pipeline_dreamzero_parity.py` 对比的是 DreamZero eager reference。结果为：`TP=1, CF_P=1/2` 严格对齐，`TP=2, CF_P=1/2` 首个失败点稳定为 `state.positive.kv[1]`，`max_diff=1.562e-02`。
- `scheduling_flow_unipc_multistep.py`：已恢复到原仓 eager-aligned 版本；新增 `tests/dreamzero/test_flow_unipc_scheduler_precision.py`，确认当前 vLLM scheduler 与 DreamZero 不加 `torch.compile` 的 scheduler 在 step0/step1 上完全一致。
- 训练路径：按要求删除，不再列为 TODO。

## 已完成

| 组件 | 当前实现 | 状态 | 验证 |
|------|---------|------|------|
| `SinusoidalPositionalEncoding` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_sinusoidal_positional_encoding` |
| `CategorySpecificLinear` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_category_specific_linear` |
| `CategorySpecificMLP` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_category_specific_mlp` |
| `MultiEmbodimentActionEncoder` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_multi_embodiment_action_encoder` |
| `DistributedRMSNorm` | `modeling/causal_wan_model.py` | 替代 DreamZero `WanRMSNorm`，支持 TP 全局 RMS | `tests/dreamzero/test_causal_wan_model.py::test_distributed_rmsnorm_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `MLPProj` | `modeling/causal_wan_model.py` | 已复用 `ColumnParallelLinear + RowParallelLinear` | 已纳入 full model 精度覆盖 |
| `WanT2VCrossAttention` | `modeling/causal_wan_model.py` | 已复用 `Attention + Column/RowParallelLinear`，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_t2v_cross_attn_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `WanI2VCrossAttention` | `modeling/causal_wan_model.py` | 已复用 `Attention + Column/RowParallelLinear`，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_i2v_cross_attn_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| cross-attn cache 语义 | `modeling/causal_wan_model.py` | 模块级 `WanT2VCrossAttention` / `WanI2VCrossAttention` 已验证 cache fill/reuse；full-model 链路按上游语义保持 uninitialized | `tests/dreamzero/test_causal_wan_model.py::test_t2v_cross_attn_cache_fill_and_reuse` / `tests/dreamzero/test_causal_wan_model.py::test_i2v_cross_attn_cache_fill_and_reuse` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `CausalWanSelfAttention` | `modeling/causal_wan_model.py` | 已复用 `Attention + Column/RowParallelLinear + DistributedRMSNorm`，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_self_attn_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `CausalWanAttentionBlock` | `modeling/causal_wan_model.py` | FFN 已切到 TP 线性层，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_attention_block_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `CausalWanModel.__init__` | `modeling/causal_wan_model.py` | `patch_embedding` 已切到 `Conv3dLayer` | `tests/dreamzero/test_causal_wan_model.py::test_hotpath_layer_types` |
| `img_emb` 创建条件 / `init_weights()` | `modeling/causal_wan_model.py` | 已与 DreamZero 构造逻辑对齐 | `tests/dreamzero/test_causal_wan_model.py::test_img_emb_created_only_for_i2v` / `tests/dreamzero/test_causal_wan_model.py::test_init_weights_called_and_matches_upstream_scheme` |
| `_create_freqs` / `unpatchify` / `_forward_blocks` / `_forward_inference` | `modeling/causal_wan_model.py` | 与 DreamZero inference 分支对齐 | `tests/dreamzero/test_causal_wan_model.py::test_full_model_precision_prefill_and_ar_step` |
| tiny full model prefill + AR step | `modeling/causal_wan_model.py` | 单卡与 TP=2 GPU 数值对齐通过 | `tests/dreamzero/test_causal_wan_model.py::test_full_model_precision_prefill_and_ar_step` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |

### TP=2 修复记录

| 项目 | 状态 | 备注 |
|------|------|------|
| TP=2 测试脚本 | 已完成 | `tests/dreamzero/test_causal_wan_model_tp2.py` 已覆盖热点层、RMSNorm、T2V/I2V、self-attn、block、full model |
| TP=1 vs TP=2 自一致性脚本 | 已完成 | `tests/dreamzero/test_causal_wan_model_tp1_vs_tp2.py`，但该脚本当前跑的是 `fp32` 路径，不能覆盖真实 `bf16` pipeline 漂移 |
| `ensure_stream` workaround | 已删除 | `.venv` 下删除后 TP=1 vs TP=2 仍完全对齐，不再保留默认 stream 恢复逻辑 |
| 原生 `RowParallelLinear` 精度表征 | 已完成 | `tests/diffusion/layers/test_row_parallel_linear_precision.py`：`TP=2 + fp32` max diff `3.814697e-06`；`TP=2 + bf16` max diff `1.250000e-01` |

## 剩余 TODO

### 性能优化

| 项目 | 现状 | 备注 |
|------|------|------|
| DiT cache skip (static mask) | 未接入 | `dit_step_mask` 跳步，8/16 步只跑模型 8 次，原始 L201-218 |
| DiT cache skip (dynamic cosine) | 未接入 | `should_run_model` 基于 cosine similarity 动态跳步，原始 L899-927 |
| Async pipeline (predicted frame feedback) | 未接入 | 用上一步 `video_pred` 代替 VAE encode 真实观测帧，省 VAE 延迟。代码已有 `latent_video` 参数透传，当前未启用。原始 L1013-1017 |

### 并行框架复用

| 项目 | 现状 | 备注 |
|------|------|------|
| Sequence Parallel / Ulysses / Ring | 未接入 | 当前注意力显式 `skip_sequence_parallel=True`，模型也没有 `_sp_plan` |
| Bagel / WAN2.2 风格的并行边界切分 | 未接入 | 需要单独设计 DreamZero 的 sharding 边界和 cache 语义 |

### 严格一致性收尾

| 项目 | 现状 | 备注 |
|------|------|------|
| `RowParallelLinear` 在 `bf16 + TP>1` 下的数值漂移 | 未解决 | 当前保留 vLLM 原生 `bf16 x bf16` 路径；专门测试 `tests/diffusion/layers/test_row_parallel_linear_precision.py` 显示单层 `max_diff=1.25e-01`，pipeline parity 首个失败点约 `1.562e-02`（`state.positive.kv[1]`） |
| DreamZero scheduler `eager` / `compiled` `bf16` 差异 | 已定位，记录现象 | 当前实现只对齐 DreamZero 不加 `torch.compile` 的 scheduler；若未来要对齐 upstream compiled 路径，需要单独处理这部分数值差异 |

### Pipeline / Serving / Transform

| 文件 | 对应 DreamZero 原始 | 状态 |
|------|-------------------|------|
| `pipeline_dreamzero.py` | `WANPolicyHead.lazy_joint_video_action` L929-1270 | 已实现，且 `tests/dreamzero/test_pipeline_dreamzero_parity.py` 已覆盖 TP=1/2、CF_P=1/2 与 upstream eager reference 对照 |
| `state_dreamzero.py` | `ARDroidRoboarenaPolicy._frame_buffers` + `WANPolicyHead.kv_cache*` | 已实现并接线；`accumulate_frames()` 已在 pipeline `forward()` 中实际参与 AR 轨迹 |
| `transform/base.py` | `dreamzero_cotrain.py` 基类 | 已实现：纯数据集关注点，3 个抽象方法 |
| `transform/droid.py` | `dreamzero_cotrain.py` OXE_DROID | 已实现：3 路相机拼接 + 语言模板 + state 提取 |
| `transform/roboarena.py` | `socket_test_optimized_AR.py` key 映射 | 已实现：继承 DroidTransform，只改 IMAGE_KEY_MAP |
| `openpi_connection.py` | `websocket_policy_server.py` | 已实现 |
| `openpi_serving.py` | 业务层 | 已实现 |
| CLIP image encoder | `wan_video_image_encoder.py` L856-891 | ✅ 复用 `CLIPVisionModel` + `CLIPImageProcessor`（wan2_2 模式），无需移植 908 行自定义代码 |
| registry 注册 | `registry.py` | 待补 |
| 端到端服务测试 | server + client | 待补 |

## 当前 vllm-omni 复用情况

| 基础设施 | 用途 |
|---------|------|
| `Conv3dLayer` | `patch_embedding` |
| `ColumnParallelLinear` | self-attn / cross-attn QKV、FFN up、`MLPProj` |
| `RowParallelLinear` | self-attn / cross-attn O、FFN down、`MLPProj` |
| `DistributedRMSNorm` | self-attn / cross-attn QK norm |
| `Attention` | self-attn / cross-attn kernel 调度与后端选择 |

## 验证命令

- `MASTER_PORT=29601 PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_action_encoder.py -v -s`
- `PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_causal_wan_model.py -v -s`
- `PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest tests/dreamzero/test_causal_wan_model_tp2.py -v -s`
- `PYTHONPATH=. .venv/bin/python tests/dreamzero/test_causal_wan_model_tp1_vs_tp2.py`
- `PYTHONPATH=. .venv/bin/python -m pytest tests/dreamzero/test_pipeline_dreamzero_parity.py -q`
- `PYTHONPATH=. .venv/bin/python -m pytest tests/dreamzero/test_flow_unipc_scheduler_precision.py -q`

## Appendix: Precision Report

### 环境与口径

- 口径：保留 vLLM 原生 `RowParallelLinear` 的 `bf16 x bf16` GEMM + TP all-reduce 路径，不再使用 DreamZero 局部 `fp32` 特化。
- 环境：`/home/yangshen/code/vllm-omni-wm`，使用 `.venv` 运行。

### 专项层级测试：`RowParallelLinear`

- 命令：`PYTHONPATH=. .venv/bin/python -m pytest tests/diffusion/layers/test_row_parallel_linear_precision.py -v -s`
- 测试文件：`tests/diffusion/layers/test_row_parallel_linear_precision.py`
- 结果：
  - `TP=2 + fp32`：`max_diff = 3.814697e-06`
  - `TP=2 + bf16`：`max_diff = 1.250000e-01`
- 结论：原生 `RowParallelLinear` 在 `bf16 + TP>1` 下，相对未切分 `F.linear` 会出现明显数值漂移；该问题在层级测试即可复现，不依赖 DreamZero pipeline。

### 端到端测试：DreamZero pipeline parity

- 命令：`PYTHONPATH=. .venv/bin/python -m pytest tests/dreamzero/test_pipeline_dreamzero_parity.py -v -s`
- 测试文件：`tests/dreamzero/test_pipeline_dreamzero_parity.py`
- 结果：
  - `tp=1, cfg_p=1`：`output_max=0.000e+00`，`state_max=0.000e+00`
  - `tp=1, cfg_p=2`：`output_max=0.000e+00`，`state_max=0.000e+00`
  - `tp=2, cfg_p=1`：首个失败点 `state.positive.kv[1]`，`max_diff=1.562e-02`
  - `tp=2, cfg_p=2`：首个失败点 `state.positive.kv[1]`，`max_diff=1.562e-02`
- 结论：保留原生 `bf16 x bf16` 的 `RowParallelLinear` 后，DreamZero pipeline 在 `TP=1` 时可与参考严格对齐，但在 `TP=2` 时会首先体现为 KV cache 漂移。

### Scheduler 精度记录：当前实现对齐 eager，compile 差异单独记录

- 测试文件：`tests/dreamzero/test_flow_unipc_scheduler_precision.py`
- 当前目标：
  - `vllm_omni` 当前 scheduler == DreamZero 不加 `torch.compile` 的 scheduler
  - DreamZero upstream `eager` vs `compiled` 差异只做记录，不作为这次端口的对齐目标
- 实测：
  - `vLLM baseline vs upstream eager`：`step0 = 0.0`，`step1 = 0.0`
  - `upstream eager vs compiled`：`step0 = 1.562500e-02`，`step1 = 1.562500e-02`
- 结果：
  - `step0_max_diff = 1.562500e-02`
  - `step1_max_diff = 1.562500e-02`
- 最小化复现（CUDA `bf16`）：
  - 仅 `a / b * x`：`mul_diff = 1.562500e-02`
  - `a / b * x - c * m`：`affine_diff = 1.562500e-02`
  - 同样表达式换成 `float32` 输入后：`mul32_diff = 0`，`affine32_diff = 2.384186e-07`
- 结论：
  - 当前仓内 scheduler 不需要为这次任务保留额外的 compiled 特化；
  - 若以后要对齐 DreamZero upstream compiled 路径，分歧点已经定位到 predictor 内部的 CUDA `bf16` 元素级乘加舍入，而不是 pipeline 接线问题。
