# 6. 精度对齐与测试

## 测试体系

6 个测试文件，22 个测试用例，覆盖：单元测试、精度对齐、协议兼容性。

```
tests/dreamzero/
├── test_video_action_scheduler.py     # 组合调度器 (2 tests)
├── test_action_encoder.py             # 动作编解码器 (4 tests)
├── test_combine_cfg_noise.py          # CFG 逻辑 (1 test)
├── test_causal_wan_model.py           # DiT 模型 (4 tests)
├── test_precision_alignment.py        # 精度对齐 (5 tests)
├── test_websocket_integration.py      # WebSocket 协议 (5 tests)
└── test_websocket_client.py           # E2E 客户端工具
```

## 精度对齐方法

### 原则

vllm-omni 的每个组件必须与 `third_party/dreamzero` 原始代码在**相同权重 + 相同输入**下产出 **bit-identical** 输出。

### 实现

```python
# test_precision_alignment.py 的核心模式：
def test_xxx_alignment():
    from vllm_omni...  import VllmComponent
    from groot.vla...  import DreamZeroComponent

    # 1. 创建两个实例
    vllm_obj = VllmComponent(**same_config)
    dz_obj = DreamZeroComponent(**same_config)

    # 2. 复制权重（确保参数一致）
    dz_obj.load_state_dict(vllm_obj.state_dict())

    # 3. 相同输入
    torch.manual_seed(42)
    x = torch.randn(...)

    # 4. 对比输出
    vllm_out = vllm_obj(x)
    dz_out = dz_obj(x)
    assert torch.allclose(vllm_out, dz_out, atol=1e-6)
```

### 已验证组件

| 组件 | 测试 | max_diff | 状态 |
|------|------|----------|------|
| `rope_params` (d=16,32,64,128) | 频率表生成 | **0.00e+00** | bit-identical |
| `causal_rope_action_apply` | 推理模式 RoPE | **0.00e+00** | bit-identical |
| `rope_action_apply` | 训练模式 RoPE | **0.00e+00** | bit-identical |
| `MultiEmbodimentActionEncoder` | 动作编码器 | **0.00e+00** | bit-identical |
| `CategorySpecificMLP` | 状态编码/动作解码 | **0.00e+00** | bit-identical |

## CausalWanModel 功能测试

不做精度对齐（需要完整模型权重），而是验证功能正确性：

| 测试 | 验证内容 | 结果 |
|------|---------|------|
| `test_causal_wan_model_init` | 模型构建，参数数量 | 162,584 params, 2 blocks |
| `test_prefill_no_action` | 首帧 prefill，无 action | KV cache 0→4 |
| `test_inference_with_action` | 去噪步，带 action | video + action 输出 |
| `test_kv_cache_grows_across_steps` | 3 步 AR，KV cache 增长 | [4, 8, 12] 单调递增 |

## WebSocket 协议测试

使用 FastAPI `TestClient` + 自包含 mini handler（避免触发完整 import 链）：

| 测试 | 验证内容 |
|------|---------|
| `test_metadata_matches_policy_server_config` | metadata 字段兼容 `PolicyServerConfig(**metadata)` |
| `test_reset_returns_string` | reset 返回 plain string `"reset successful"` |
| `test_infer_returns_raw_ndarray` | infer 返回 `msgpack(ndarray)`，shape=(24,8) |
| `test_multi_round_like_test_client_ar` | 1 init + 3 chunks + reset 全流程 |
| `test_msgpack_numpy_roundtrip` | ndarray 序列化/反序列化保真 |

## Bug 修复记录

### RoPE action_state_index 负数

**问题**：`current_start_frame=0` 时，`action_state_index = (0-1)//1 = -1`，导致 `freqs_action[-4:0]` 为空张量。

**修复**：`action_state_index = max(0, (current_start_frame - 1) // num_frame_per_block)`

**根因**：首帧 prefill 不应有 action token（DreamZero 原始代码在 prefill 时传 `action=None`）。

### rope_action_apply 零长 state

**问题**：`num_state_per_block=0` 时 `freqs_state[:0].view(0, 1, -1)` 无法推断最后维度。

**修复**：guard `n_state > 0` 再 view。

## 运行测试

```bash
PYTHONPATH=/path/to/vllm-omni-wm:$PYTHONPATH \
  python tests/dreamzero/test_video_action_scheduler.py && \
  python tests/dreamzero/test_action_encoder.py && \
  python tests/dreamzero/test_combine_cfg_noise.py && \
  python tests/dreamzero/test_causal_wan_model.py && \
  python tests/dreamzero/test_precision_alignment.py && \
  python tests/dreamzero/test_websocket_integration.py
```

期望输出：`6/6 ALL TESTS PASSED`

## CI 状态

| Check | 状态 |
|-------|------|
| DCO | ✅ pass |
| pre-commit | ✅ pass |
| build (3.11) | ✅ pass |
| build (3.12) | ✅ pass |
