# DreamZero Implementation TODO

## 已完成

| 组件 | 文件 | 精度测试结果 | 测试位置 |
|------|------|------------|---------|
| `sinusoidal_embedding_1d` | `modeling/causal_wan_model.py` | 0.00e+00 bit-identical | `tests/dreamzero/test_causal_wan_model.py::test_rope_precision` |
| `rope_params` | `modeling/causal_wan_model.py` | 0.00e+00 bit-identical (d=16,32,64,128) | 同��� |
| `rope_apply` | `modeling/causal_wan_model.py` | 0.00e+00 bit-identical | 同上 |
| `rope_action_apply` | `modeling/causal_wan_model.py` | 0.00e+00 bit-identical | 同上 |
| `causal_rope_action_apply` | `modeling/causal_wan_model.py` | 0.00e+00 bit-identical | 同上 |
| `WanRMSNorm` | `modeling/causal_wan_model.py` (import vllm RMSNorm) | 0.00e+00 bit-identical (dim=64,128,5120) | `tests/dreamzero/test_causal_wan_model.py::test_rmsnorm_precision` L245 |
| `WanLayerNorm` | `modeling/causal_wan_model.py` | — (nn.LayerNorm 封装) | — |
| `MLPProj` | `modeling/causal_wan_model.py` | 0.00e+00 bit-identical | `tests/dreamzero/test_causal_wan_model.py::test_mlpproj_precision` L267 |
| `SinusoidalPositionalEncoding` | `modeling/action_encoder.py` | 0.00e+00 bit-identical (dim=64,128,256,5120) | `tests/dreamzero/test_action_encoder.py::test_sinusoidal_positional_encoding` |
| `CategorySpecificLinear` | `modeling/action_encoder.py` | 0.00e+00 bit-identical | `tests/dreamzero/test_action_encoder.py::test_category_specific_linear` |
| `CategorySpecificMLP` | `modeling/action_encoder.py` | 0.00e+00 bit-identical | `tests/dreamzero/test_action_encoder.py::test_category_specific_mlp` |
| `MultiEmbodimentActionEncoder` | `modeling/action_encoder.py` | 0.00e+00 bit-identical | `tests/dreamzero/test_action_encoder.py::test_multi_embodiment_action_encoder` |
| `DreamZeroState.reset` | `state_dreamzero.py` | — (纯状态管理) | — |
| `DreamZeroState.accumulate_frames` | `state_dreamzero.py` | 待测 | 待写 |
| `DreamZeroState.should_reset` | `state_dreamzero.py` | 待测 | 待写 |
| `DreamZeroState.create_kv_caches` | `state_dreamzero.py` | 待测 | 待写 |
| `DreamZeroState.update_kv_cache` | `state_dreamzero.py` | 待测 | 待写 |
| `DreamZeroState.get_kv_caches` | `state_dreamzero.py` | 待测 | 待写 |
| `ServingRealtimeRobotOpenPI` | `entrypoints/.../openpi_serving.py` | — (serving 层) | — |
| `RobotRealtimeConnection` | `entrypoints/.../openpi_connection.py` | — (协议层) | — |
| Transform (base/droid/roboarena) | `entrypoints/.../transform/` | — (key 映射) | — |

## causal_wan_model.py 待实现

### 需要修改已有代码

| 组件 | 当前实现 | 改成 | 精度测试结果 | 测试位置 |
|------|---------|------|------------|---------|
| `MLPProj` | `nn.Linear` × 2 | `ColumnParallelLinear` + `RowParallelLinear` | 待重测 | 待更新 |

### 需要新实现

| 组件 | 复用 vllm 算子 | 对应 DreamZero 原始 | 精度测试结果 | 测试位置 |
|------|--------------|-------------------|------------|---------|
| `CausalWanSelfAttention.__init__` | `QKVParallelLinear` + `RowParallelLinear` + `RMSNorm` + `Attention(causal)` | `wan_video_dit...py` L188-224 | 待测 | 待写 |
| `CausalWanSelfAttention.forward` (KV cache path) | `Attention` | `wan_video_dit...py` L1008-1084 | 待测 | 待写 |
| `WanT2VCrossAttention` | `QKVParallelLinear` + `RowParallelLinear` + `Attention` | `wan2_1_submodule.py` L243-306 | 待测 | 待写 |
| `WanI2VCrossAttention` | 同上 | `wan2_1_submodule.py` L308-362 | 待测 | 待写 |
| `CausalWanAttentionBlock.__init__` | 组合 self-attn + cross-attn + ffn (`ColumnParallelLinear` + `RowParallelLinear`) | `wan_video_dit...py` L1087-1141 | 待测 | 待写 |
| `CausalWanAttentionBlock.forward` | — | `wan_video_dit...py` L1142-1187 | 待测 | 待写 |
| `CausalHead.__init__` | `nn.Linear` (跑一次不 TP) | `wan_video_dit...py` L1190-1205 | 待测 | 待写 |
| `CausalHead.forward` | — | `wan_video_dit...py` L1207-1215 | 待测 | 待写 |
| `CausalWanModel.__init__` | `Conv3dLayer` + 组装所有组件 | `wan_video_dit...py` L1230-1387 | 待测 | `tests/dreamzero/test_causal_wan_model.py::test_init` |
| `CausalWanModel._create_freqs` | — | `wan_video_dit...py` L2151-2174 | 待测 | 待写 |
| `CausalWanModel.unpatchify` | — | `wan_video_dit...py` L2127-2149 | 待测 | 待写 |
| `CausalWanModel._forward_blocks` | — | `wan_video_dit...py` L1691-1779 | 待测 | 待写 |
| `CausalWanModel._forward_inference` | — | `wan_video_dit...py` L1863-1950 | 待测 | `tests/dreamzero/test_causal_wan_model.py::test_prefill` / `test_inference_with_action` / `test_kv_cache_growth` |

## 其他待实现文件

| 文件 | 对应 DreamZero 原始 | 精度测试结果 | 测试位置 |
|------|-------------------|------------|---------|
| `pipeline_dreamzero.py` | `WANPolicyHead.lazy_joint_video_action` L929-1270 | 待测 | 待写 |
| `modeling/wan_video_image_encoder.py` | `wan_video_image_encoder.py` L856-891 | 待测 | 待写 |
| `action_transform.py` | `GrootSimPolicy.apply/unapply` | 待测 | 待写 |
| registry 注册 | `registry.py` 加一行 | — | — |
| 端到端测试 | server + test_client_AR.py | 待测 | 待写 |

## vllm 算子复用汇总

| vllm 算子 | import 路径 | 用在哪 |
|-----------|-----------|--------|
| `QKVParallelLinear` | `vllm.model_executor.layers.linear` | self-attn Q/K/V (×40), cross-attn Q/K/V (×40) |
| `RowParallelLinear` | 同上 | self-attn O (×40), cross-attn O (×40), FFN down (×40), MLPProj down |
| `ColumnParallelLinear` | 同上 | FFN up (×40), MLPProj up |
| `RMSNorm` | `vllm.model_executor.layers.layernorm` | QK norm (×80) |
| `Attention` | `vllm_omni.diffusion.attention.layer` | self-attn (×40), cross-attn (×40) |
| `Conv3dLayer` | `vllm.model_executor.layers.conv` | patch_embedding (×1) |

## 不复用的组件

| 组件 | 原因 |
|------|------|
| `text_embedding` (nn.Linear×2) | 跑一次，不在热路径 |
| `time_embedding` (nn.Linear×2) | 同上 |
| `time_projection` (nn.Linear×1) | 同上 |
| `CausalHead` (nn.Linear×1) | 同上 |
| `WanLayerNorm` (nn.LayerNorm) | 标准 PyTorch |
| `CategorySpecificLinear/MLP` | 非标准 Linear (per-embodiment weights)，不能替换 |
| CLIP (WanImageEncoder) | open_clip，与 vllm 的 HF CLIP 不同 |
