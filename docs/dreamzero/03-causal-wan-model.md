# 3. CausalWanModel：40 层 DiT 的因果注意力与 KV Cache

文件：`vllm_omni/diffusion/models/dreamzero/modeling/causal_wan_model.py`（1983 行）

## 与 WanTransformer3DModel 的关键差异

| 特性 | WanTransformer3DModel (Wan2.2) | CausalWanModel (DreamZero) |
|------|------|------|
| 注意力 | 全局注意力（所有 token 互相可见） | **因果注意力**（新帧只看历史帧） |
| KV Cache | 无 | **有**（per-layer，流式推理增量更新） |
| 输出 | 单输出（video noise pred） | **双输出**（video + action noise pred） |
| 额外 token | 无 | **action token + state token** 拼接在视频序列后 |
| RoPE | 标准 3D RoPE | 扩展 RoPE（action/state 有独立频率） |
| Forward 模式 | 单一 | **两种**：inference（KV cache）/ train（teacher forcing） |

## 模型结构

```
CausalWanModel
├── patch_embedding: Conv3d(16, 5120, kernel=(1,2,2), stride=(1,2,2))
├── text_embedding: Linear(4096, 5120) → GELU → Linear(5120, 5120)
├── time_embedding: Linear(256, 5120) → SiLU → Linear(5120, 5120)
├── time_projection: SiLU → Linear(5120, 5120×6)
├── img_emb: MLPProj(1280, 5120)   # CLIP 特征投影 (i2v 模式)
├── action_encoder: MultiEmbodimentActionEncoder
├── state_encoder: CategorySpecificMLP
├── action_decoder: CategorySpecificMLP
├── blocks: 40 × CausalWanAttentionBlock
│   ├── self_attn: CausalWanSelfAttention
│   │   ├── q, k, v: Linear(5120, 5120)
│   │   ├── o: Linear(5120, 5120)
│   │   ├── norm_q, norm_k: WanRMSNorm
│   │   └── 因果注意力 + KV cache 逻辑
│   ├── cross_attn: WanI2VCrossAttention / WanT2VCrossAttention
│   └── ffn: Linear(5120, 13824) → GELU → Linear(13824, 5120)
├── head: CausalHead (LayerNorm → Linear → unpatchify)
├── freqs: 3 组空间 RoPE 频率
├── freqs_action: action 专用 RoPE
└── freqs_state: state 专用 RoPE
```

## KV Cache 机制

### 创建

```python
# per-layer cache: [2, batch, 0, num_heads, head_dim]
# dim 0: [Key, Value]
# dim 2: seq_len（从 0 开始，每步追加）
kv_cache = [torch.zeros(2, B, 0, 40, 128) for _ in range(40)]
```

### 推理时更新

```python
# CausalWanSelfAttention.forward() 中：
# 1. 计算当前 token 的 Q, K, V
# 2. 将新 K, V 追加到 cache
new_kv = torch.cat([kv_cache, torch.stack([new_k, new_v])], dim=2)
# 3. 用完整 cache 的 K, V 做注意力
attn_out = F.scaled_dot_product_attention(q, new_kv[0], new_kv[1])
# 4. 返回更新后的 cache
```

### KV Cache 增长示例

```
Step 0 (prefill):  seq_len = 0 → 4   (首帧 4 个 patch token)
Step 1 (denoise):  seq_len = 4 → 8   (第二帧 + action/state token)
Step 2 (denoise):  seq_len = 8 → 12  (第三帧 + action/state token)
```

测试验证：`test_causal_wan_model.py::test_kv_cache_grows_across_steps` → `[4, 8, 12]`

## RoPE 扩展

标准 3D RoPE（空间 T×H×W）之外，DreamZero 为 action 和 state token 分配独立的 RoPE 频率：

```python
d = dim // num_heads  # = 128
freqs_action = rope_params(10240, d)  # 10240 是 action 的最大位置
freqs_state = rope_params(1024, d)    # 1024 是 state 的最大位置
freqs = [
    rope_params(1024, d - 4*(d//6)),  # 时间维 = 44 complex
    rope_params(1024, 2*(d//6)),      # 高度维 = 42 complex
    rope_params(1024, 2*(d//6)),      # 宽度维 = 42 complex
]
# 三维 concat 后 = 44+42+42 = 128/2 = 64 complex = 128 real = d
```

`causal_rope_action_apply()`：推理时，将当前帧的 action/state 频率拼接到空间频率后面。

## 因果注意力设计

### 训练模式（teacher forcing）

```
输入序列: [clean_frames | noisy_frames | action_tokens | state_tokens]

注意力规则:
- clean_frame[i] 可以看: clean_frame[0..i] (因果)
- noisy_frame[i] 可以看: 所有 clean + noisy frame[0..i] (双向看clean，因果看noisy)
- action_token[i] 可以看: 所有 clean + noisy + action[0..i] + state[i]
- state_token[i] 只能自注意力
```

### 推理模式（KV cache）

```
每步只处理 1 帧:
- 新 token 查询完整 KV cache（包含所有历史帧）
- 因果性由 KV cache 自然保证（只有历史帧在 cache 中）
- action/state token 的 RoPE 由 action_state_index 确定位置
```

## Forward 路由

```python
def forward(self, x, timestep, kv_cache=None, ...):
    if kv_cache is not None:
        return self._forward_inference(...)  # KV cache 模式
    else:
        return self._forward_train(...)      # 全序列模式
```

## 精度对齐验证

| 组件 | max_diff | 状态 |
|------|----------|------|
| `rope_params` | 0.00e+00 | bit-identical |
| `causal_rope_action_apply` | 0.00e+00 | bit-identical |
| `rope_action_apply` | 0.00e+00 | bit-identical |
