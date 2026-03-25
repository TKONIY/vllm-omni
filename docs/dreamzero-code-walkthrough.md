# DreamZero Pipeline 代码解读

本文档解读 DreamZero 在 vllm-omni 中的完整实现，覆盖架构设计、端到端执行流程和核心组件实现细节。

## 1. 架构总览

DreamZero 是一个**联合视频-动作扩散模型**，基于 Wan2.1-I2V-14B（140 亿参数 DiT），一次推理同时输出视频帧和机器人动作（7 关节 + 1 夹爪 = 8 维 × 24 步）。

### 请求处理全路径

```
WebSocket Client (test_client_AR.py)
  │ msgpack binary
  ▼
/v1/world/openpi ─── OmniWorldStreamHandler
  │                     ├─ WorldSessionStore (会话管理)
  │                     └─ 构建 OmniDiffusionRequest
  ▼
DreamZeroPipeline.forward()
  ├─ _encode_prompt()        → UMT5-XXL 文本编码
  ├─ VAE encode              → 图像 → latent
  ├─ _prefill_kv_cache()     → 首帧编码进 KV cache
  ├─ diffuse()               → 去噪循环 (16 步)
  │   ├─ predict_noise_maybe_with_cfg()
  │   │   ├─ rank 0: CausalWanModel(cond)
  │   │   ├─ rank 1: CausalWanModel(uncond)
  │   │   ├─ all_gather
  │   │   └─ combine: 视频CFG + 动作positive-only
  │   ├─ scheduler_step_maybe_with_cfg()
  │   │   └─ VideoActionScheduler → FlowUniPC×2
  │   └─ cuda.synchronize()
  └─ return actions (24, 8)
```

### 文件结构

```
vllm_omni/diffusion/models/dreamzero/
├── pipeline_dreamzero.py              # 主 pipeline (596 行)
│   ├── VideoActionScheduler           # 组合调度器 (~20 行)
│   └── DreamZeroPipeline              # CFGParallelMixin 集成
└── modeling/
    ├── causal_wan_model.py            # 40 层 DiT (1983 行)
    └── action_encoder.py              # 动作编解码 (164 行)

vllm_omni/entrypoints/openai/
├── serving_world_stream.py            # WebSocket handler
├── session_manager.py                 # 会话状态
└── protocol/world.py                  # 协议定义
```

### 组件复用

| 组件 | 来源 | 说明 |
|------|------|------|
| UMT5-XXL | Wan2.2 复用 | `from_pretrained` 直接加载 |
| VAE | Wan2.2 复用 | `DistributedAutoencoderKLWan` |
| FlowUniPC Scheduler | 公共复用 | 视频和动作各一个实例 |
| CFGParallelMixin | 公共复用 | all_gather + 本地 combine |
| CausalWanModel | **新写** | 因果注意力 + KV cache |
| ActionEncoder | **新写** | 按机器人类型选择权重 |

---

## 2. DreamZeroPipeline（pipeline_dreamzero.py）

### 2.1 VideoActionScheduler

跟 PR #2160 (LTX2 VideoAudioScheduler) 相同模式，将两个独立 scheduler 封装为一个：

```python
class VideoActionScheduler:
    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(noise_pred[0], t[0], latents[0], ...)[0]
        action_out = self.action_scheduler.step(noise_pred[1], t[1], latents[1], ...)[0]
        return ((video_out, action_out),)
```

`scheduler_step_maybe_with_cfg()` 通过 `per_request_scheduler=video_action_scheduler` 调用。

### 2.2 combine_cfg_noise — 视频 CFG + 动作 Positive-only

```python
def combine_cfg_noise(self, pos, neg, scale, normalize):
    (video_pos, action_pos) = pos
    (video_neg, _) = neg
    video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
    return (video_combined, action_pos)  # 动作不做 CFG
```

**为什么动作不做 CFG？** 动作应直接跟随语言指令，无条件分支的"漫无目的"动作没有意义。

### 2.3 KV Cache 管理

DreamZero 是**自回归**推理，KV cache 跨调用累积：

```python
def _prefill_kv_cache(self, image_latents, prompt_embeds, ...):
    if self.current_start_frame == 0:
        # 首次：创建空 cache，编码首帧
        self.kv_cache = [zeros(2, B, 0, heads, d) for _ in range(40)]
        self.transformer(first_frame, timestep=0, kv_cache=self.kv_cache)
        # → KV cache: [2, B, 4, 40, 128]（首帧 4 个 token）
        self.current_start_frame = 1
```

每个 rank 持有自己的 cache（rank 0 = 条件分支，rank 1 = 无条件分支），mixin 保持无状态。

### 2.4 CFG Parallel 同步

```python
def _synchronize_cfg_parallel_step_output(self, latents, do_true_cfg):
    latents = tuple(t.contiguous() for t in latents)
    if self._is_cfg_parallel_enabled(do_true_cfg):
        torch.cuda.current_stream(device).synchronize()
    return latents
```

`.contiguous()` + `cuda.synchronize()` 确保 all_gather 后两个 rank 的 scheduler step 结果 bit-identical。

### 2.5 forward() 完整流程

```python
def forward(self, req):
    # 1. 会话管理
    if extra_args.get("reset"): self._reset_session()
    # 2. 文本编码
    prompt_embeds, neg_embeds = self._encode_prompt(prompt)
    # 3. VAE 编码
    video_latents = self.vae.encode(obs_images)
    # 4. KV cache prefill
    self._prefill_kv_cache(video_latents, prompt_embeds, ...)
    # 5. 准备噪声
    noise_video = randn_like(video_latents)
    noise_action = randn(B, 24, 8)
    # 6. 创建调度器
    scheduler = VideoActionScheduler(FlowUniPC(), FlowUniPC())
    # 7. 去噪循环
    video_out, action_out = self.diffuse(noise_video, noise_action, ...)
    # 8. 返回
    return DiffusionOutput(custom_output={"actions": action_out.numpy()})
```

---

## 3. CausalWanModel（causal_wan_model.py, 1983 行）

### 3.1 与 WanTransformer3DModel 的差异

| | WanTransformer3DModel | CausalWanModel |
|---|---|---|
| 注意力 | 全局 | **因果**（新帧只看历史帧） |
| KV Cache | 无 | **有**（per-layer 增量更新） |
| 输出 | 单输出 | **双输出**（video + action） |
| 额外 token | 无 | action + state token |
| RoPE | 标准 3D | 扩展（action/state 独立频率） |

### 3.2 模型结构

```
CausalWanModel (dim=5120, heads=40, layers=40)
├── patch_embedding: Conv3d(16→5120, stride=(1,2,2))
├── text_embedding: Linear(4096→5120) → GELU → Linear
├── time_embedding: Linear(256→5120) → SiLU → Linear → SiLU → Linear(→5120×6)
├── img_emb: MLPProj(1280→5120)  # CLIP 投影
├── action_encoder: MultiEmbodimentActionEncoder
├── state_encoder: CategorySpecificMLP
├── action_decoder: CategorySpecificMLP
├── blocks × 40:
│   ├── CausalWanSelfAttention (Q/K/V + RoPE + KV cache)
│   ├── WanI2VCrossAttention (文本+图像交叉注意力)
│   └── FFN: Linear(5120→13824) → GELU → Linear(→5120)
└── CausalHead: LayerNorm → Linear → unpatchify
```

### 3.3 KV Cache 机制

```python
# 创建：每层 [2(K,V), B, 0(空), heads, head_dim]
kv_cache = [zeros(2, B, 0, 40, 128) for _ in range(40)]

# 推理时追加：
new_kv = cat([kv_cache[layer], stack([new_k, new_v])], dim=2)
attn_out = sdpa(q, new_kv[0], new_kv[1])

# 增长验证（测试输出）：
# Step 0: seq_len 0→4
# Step 1: seq_len 4→8
# Step 2: seq_len 8→12
```

### 3.4 RoPE 扩展

```python
d = 5120 // 40 = 128  # head_dim
freqs_action = rope_params(10240, 128)  # action 独立频率
freqs_state  = rope_params(1024, 128)   # state 独立频率
freqs = [                               # 3D 空间频率
    rope_params(1024, 44),  # 时间
    rope_params(1024, 42),  # 高度
    rope_params(1024, 42),  # 宽度
]  # 44+42+42 = 128/2 complex = 128 real = d
```

推理时 `causal_rope_action_apply()` 将当前帧的 action/state 频率拼接到空间频率后。

### 3.5 Forward 路由

```python
def forward(self, x, ..., kv_cache=None, ...):
    if kv_cache is not None:
        return self._forward_inference(...)  # 流式推理
    else:
        return self._forward_train(...)      # 全序列训练
```

---

## 4. Action Encoder（action_encoder.py, 164 行）

### CategorySpecificLinear — 按机器人类型选权重

```python
class CategorySpecificLinear(nn.Module):
    # W: [num_embodiments, in_dim, out_dim] — 每种机器人一套权重
    # forward: selected_W = W[embodiment_id]; out = bmm(x, selected_W) + b
```

### MultiEmbodimentActionEncoder

```
action (B,24,8) → CategorySpecificLinear → SiLU
                                            ↓
timestep (B,24) → SinusoidalPosEnc → concat → CategorySpecificLinear → SiLU
                                                                         ↓
                                               CategorySpecificLinear → output (B,24,5120)
```

---

## 5. WebSocket Serving（serving_world_stream.py）

### 协议（完全兼容 DreamZero test_client_AR.py）

| 消息 | 格式 | 说明 |
|------|------|------|
| 连接后 metadata | `msgpack(PolicyServerConfig fields)` | n_external_cameras, action_space 等 |
| infer response | `msgpack(ndarray)` | **直接发 raw ndarray**，不是 dict |
| reset response | plain string `"reset successful"` | **不是** msgpack |

### 会话生命周期

```
新 session_id → needs_reset=True → KV cache 初始化
同 session_id → needs_reset=False → KV cache 继续
endpoint=reset → 清空 cache → 重新初始化
disconnect → 销毁会话
TTL 300s → 自动过期
```

---

## 6. 精度验证

### bit-identical 对齐（与 DreamZero 原始代码对比）

| 组件 | max_diff |
|------|----------|
| rope_params (d=16,32,64,128) | **0.00e+00** |
| causal_rope_action_apply | **0.00e+00** |
| rope_action_apply | **0.00e+00** |
| MultiEmbodimentActionEncoder | **0.00e+00** |
| CategorySpecificMLP | **0.00e+00** |

### 测试矩阵

```
✅ test_video_action_scheduler — 组合调度器 step + generator
✅ test_action_encoder — 4 个子组件
✅ test_combine_cfg_noise — CFG 逻辑
✅ test_causal_wan_model — init + prefill + inference + KV growth
✅ test_precision_alignment — 5 组件 bit-identical
✅ test_websocket_integration — metadata + reset + infer + multi-round + protocol

6/6 文件，22 用例，ALL PASS
```

### Bug 修复

1. **RoPE action_state_index 负数**：`current_start_frame=0` → `index=-1` → 空切片。修复：`max(0, ...)`
2. **rope_action_apply 零长 state**：`num_state_per_block=0` 时 view 失败。修复：guard `n_state > 0`
3. **pickle import 被禁**：替换为 `np.savez/np.load`
