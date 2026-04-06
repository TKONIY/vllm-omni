# DreamZero Implementation TODO

## 当前状态

- `action_encoder.py`：已完成，对齐 DreamZero，GPU 精度测试通过。
- `causal_wan_model.py`：inference-only 端口已完成，热点路径已经切到 vllm-omni 基础设施，单卡与 TP=2 GPU 对齐测试通过。
- 训练路径：按要求删除，不再列为 TODO。

## 已完成

| 组件 | 当前实现 | 状态 | 验证 |
|------|---------|------|------|
| `SinusoidalPositionalEncoding` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_sinusoidal_positional_encoding` |
| `CategorySpecificLinear` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_category_specific_linear` |
| `CategorySpecificMLP` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_category_specific_mlp` |
| `MultiEmbodimentActionEncoder` | `modeling/action_encoder.py` | 与 DreamZero 一致 | `tests/dreamzero/test_action_encoder.py::test_multi_embodiment_action_encoder` |
| `DistributedRMSNorm` | `modeling/causal_wan_model.py` | 替代 DreamZero `WanRMSNorm`，支持 TP 全局 RMS；已修复 TP=2 stream/all-reduce 精度问题 | `tests/dreamzero/test_causal_wan_model.py::test_distributed_rmsnorm_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `MLPProj` | `modeling/causal_wan_model.py` | 已复用 `ColumnParallelLinear + RowParallelLinear` | 已纳入 full model 精度覆盖 |
| `WanT2VCrossAttention` | `modeling/causal_wan_model.py` | 已复用 `Attention + Column/RowParallelLinear`，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_t2v_cross_attn_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `WanI2VCrossAttention` | `modeling/causal_wan_model.py` | 已复用 `Attention + Column/RowParallelLinear`，已修复 TP=2 I2V 偏差 | `tests/dreamzero/test_causal_wan_model.py::test_i2v_cross_attn_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `CausalWanSelfAttention` | `modeling/causal_wan_model.py` | 已复用 `Attention + Column/RowParallelLinear + DistributedRMSNorm`，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_self_attn_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `CausalWanAttentionBlock` | `modeling/causal_wan_model.py` | FFN 已切到 TP 线性层，单卡与 TP=2 对齐 | `tests/dreamzero/test_causal_wan_model.py::test_attention_block_precision` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |
| `CausalWanModel.__init__` | `modeling/causal_wan_model.py` | `patch_embedding` 已切到 `Conv3dLayer` | `tests/dreamzero/test_causal_wan_model.py::test_hotpath_layer_types` |
| `_create_freqs` / `unpatchify` / `_forward_blocks` / `_forward_inference` | `modeling/causal_wan_model.py` | 与 DreamZero inference 分支对齐 | `tests/dreamzero/test_causal_wan_model.py::test_full_model_precision_prefill_and_ar_step` |
| tiny full model prefill + AR step | `modeling/causal_wan_model.py` | 单卡与 TP=2 GPU 数值对齐通过 | `tests/dreamzero/test_causal_wan_model.py::test_full_model_precision_prefill_and_ar_step` / `tests/dreamzero/test_causal_wan_model_tp2.py::test_causal_wan_model_tp2_precision` |

### TP=2 修复记录

| 项目 | 状态 | 备注 |
|------|------|------|
| TP=2 测试脚本 | 已完成 | `tests/dreamzero/test_causal_wan_model_tp2.py` 已覆盖热点层、RMSNorm、T2V/I2V、self-attn、block、full model |
| TP=2 I2V 精度偏差 | 已修复 | 根因是 `DistributedRMSNorm` 在 TP=2 下通过 pynccl stream 规约时与本地算子发生竞态，已改为 TP group 上的 `torch.distributed.all_reduce(...)`，并在 I2V 路径补充默认 stream 恢复 |

## 剩余 TODO

### 并行框架复用

| 项目 | 现状 | 备注 |
|------|------|------|
| Sequence Parallel / Ulysses / Ring | 未接入 | 当前注意力显式 `skip_sequence_parallel=True`，模型也没有 `_sp_plan` |
| Bagel / WAN2.2 风格的并行边界切分 | 未接入 | 需要单独设计 DreamZero 的 sharding 边界和 cache 语义 |

### 严格一致性收尾

| 项目 | 现状 | 备注 |
|------|------|------|
| `img_emb` 创建条件 | 与原始不一致 | 当前 `("i2v", "ti2v")`，原始仅 `"i2v"` |
| `init_weights()` | 未保留 | 仅影响裸模型构造，不影响已加载权重的推理路径 |

### 其他未完成文件

| 文件 | 对应 DreamZero 原始 | 状态 |
|------|-------------------|------|
| `pipeline_dreamzero.py` | `WANPolicyHead.lazy_joint_video_action` L929-1270 | 待测 |
| `modeling/wan_video_image_encoder.py` | `wan_video_image_encoder.py` L856-891 | 待测 |
| `action_transform.py` | `GrootSimPolicy.apply/unapply` | 待测 |
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
