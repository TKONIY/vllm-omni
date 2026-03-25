# DreamZero Serving 端到端开销分析

## 端到端时序图

```
Client                Handler              Pipeline              GPU
  │                     │                     │                    │
  ├─ send obs ─────────►│                     │                    │
  │   (msgpack)         │                     │                    │
  │                     ├─ unpack ────────────►│                    │
  │                     │   (msgpack decode)   │                    │
  │                     │                     ├─ text encode ─────►│ ← Phase 1
  │                     │                     │   (UMT5-XXL)       │
  │                     │                     ├─ image encode ────►│ ← Phase 2
  │                     │                     │   (CLIP ViT-H/14)  │
  │                     │                     ├─ VAE encode ──────►│ ← Phase 3
  │                     │                     │   (Wan2.1 VAE)     │
  │                     │                     ├─ KV prefill ──────►│ ← Phase 4
  │                     │                     │   (DiT 1-2 fwd)    │
  │                     │                     ├─ denoise loop ────►│ ← Phase 5
  │                     │                     │   (DiT × N steps)  │
  │                     │                     ├─ scheduler step ──►│ ← Phase 6
  │                     │                     │   (FlowUniPC)      │
  │                     │◄─ actions ──────────┤                    │
  │                     │   (numpy array)      │                    │
  │◄─ send response ───┤                     │                    │
  │   (msgpack)         │                     │                    │
```

## 各阶段开销分解

### 首次调用（Session Reset）

| 阶段 | 组件 | 操作 | 典型耗时 (H100) | 占比 |
|------|------|------|-----------------|------|
| **Phase 0** | Handler | msgpack decode + obs 转换 | ~1ms | <1% |
| **Phase 1** | Text Encoder | UMT5-XXL forward (512 tokens) | ~50ms | ~5% |
| **Phase 2** | Image Encoder | CLIP ViT-H/14 (3 cameras × 180×320) | ~30ms | ~3% |
| **Phase 3** | VAE Encoder | Wan2.1 VAE encode (tiled) | ~40ms | ~4% |
| **Phase 4** | KV Prefill | DiT forward ×2 (cond + uncond 首帧) | ~150ms | ~15% |
| **Phase 5** | Denoising | DiT forward × 16 steps × 2 branches | ~650ms | ~65% |
| **Phase 6** | Scheduler | FlowUniPC step × 16 (video + action) | ~5ms | ~1% |
| **Phase 7** | Handler | action → numpy → msgpack encode + send | ~1ms | <1% |
| | | **总计** | **~930ms** | |

### 后续调用（KV Cache 已有）

| 阶段 | 操作 | 典型耗时 (H100) | 占比 |
|------|------|-----------------|------|
| **Phase 0** | msgpack decode + obs 转换 | ~1ms | <1% |
| Phase 1 | Text Encoder — **跳过**（可缓存） | 0ms | 0% |
| Phase 2 | Image Encoder — **跳过**（CLIP 特征已缓存） | 0ms | 0% |
| **Phase 3** | VAE encode（新帧 only） | ~15ms | ~2% |
| **Phase 4** | KV Prefill（追加 1 帧到 cache） | ~80ms | ~12% |
| **Phase 5** | Denoising（16 步，但 DiT cache 可跳步） | ~500ms | ~75% |
| **Phase 6** | Scheduler step | ~5ms | ~1% |
| **Phase 7** | msgpack encode + send | ~1ms | <1% |
| | **总计** | **~600ms** | |

### 使用 DiT Cache 加速（8/16 步）

| 阶段 | 操作 | 典型耗时 (H100) | 占比 |
|------|------|-----------------|------|
| Phase 3 | VAE encode | ~15ms | ~5% |
| Phase 4 | KV Prefill | ~80ms | ~25% |
| **Phase 5** | Denoising（**8 步**实际计算，8 步复用缓存） | ~280ms | ~65% |
| Phase 6 | Scheduler step | ~5ms | ~2% |
| | **总计** | **~380ms** | |

## 各阶段详细说明

### Phase 1: Text Encoder (UMT5-XXL)

```
输入: prompt string → tokenizer → [1, 512] token ids
模型: UMT5-XXL encoder (T5-XXL variant)
输出: [1, 512, 4096] text embeddings
```

- 首次调用编码 prompt，后续如果 prompt 不变可缓存
- 无条件分支需要编码空 prompt（CFG 需要）
- **优化**：prompt 不变时跳过重复编码

### Phase 2: Image Encoder (CLIP ViT-H/14)

```
输入: first frame [1, 3, H, W] → resize to (224, 224)
模型: open-clip xlm-roberta-large ViT-H/14
输出: clip_features [1, 257, 1280]（CLS + 256 patch tokens）
```

- 仅首帧需要编码（CLIP 特征在整个 session 内缓存）
- 后续调用直接复用 `self.clip_feas`

### Phase 3: VAE Encoder (Wan2.1)

```
输入: video frames [1, 3, T, H, W] → [-1, 1] normalized
模型: Wan2.1 VAE encoder (tiled for memory efficiency)
输出: latent [1, 16, T_lat, H/8, W/8]
参数: tile_size=(34,34), tile_stride=(18,16)
```

- 首次调用编码全部帧，生成 `self.ys`（首帧条件 latent）
- 后续调用仅编码新观测帧

### Phase 4: KV Cache Prefill

```
首次 (start_frame=0):
  → 创建空 KV cache: 40 层 × [2, B, 0, 40, 128]
  → DiT forward (首帧, timestep=0, no action) — 填充 cond cache
  → DiT forward (首帧, timestep=0, neg prompt) — 填充 uncond cache

后续 (start_frame>0):
  → DiT forward (新帧, timestep=0, append to cache)
```

- 是**延迟的主要贡献者**之一（尤其首次调用需要两次 forward）
- CFG parallel: 两个 rank 各持一个 cache，各跑一个分支

### Phase 5: Denoising Loop（最大开销）

```
for step in range(16):  # num_inference_steps
    # CFG parallel: rank 0 跑 cond, rank 1 跑 uncond
    video_pred, action_pred = CausalWanModel(noisy_input, kv_cache)
    # all_gather 交换结果
    # 本地 combine: video CFG + action positive-only
    # scheduler step: FlowUniPC (video + action)
```

**单步 DiT forward 耗时分解**:

| 子阶段 | 操作 | 典型耗时 |
|--------|------|----------|
| Patch Embedding | Conv3d(16→5120) | ~0.5ms |
| Time/Text Embedding | Linear + GELU + Linear | ~0.2ms |
| Self-Attention × 40 | QKV + RoPE + SDPA + O | ~25ms |
| Cross-Attention × 40 | QK(text) + V + O | ~8ms |
| FFN × 40 | Linear(5120→13824) + GELU + Linear | ~5ms |
| Output Head | LayerNorm + Linear + unpatchify | ~0.3ms |
| **单步总计** | | **~39ms** |

16 步 × 39ms = ~624ms（单 GPU）
CFG parallel (2 GPU): ~312ms + all_gather 开销

**DiT Cache 加速**:
- `dit_step_mask = [T,T,T,F,F,F,T,F,F,F,T,F,F,T,T,T]`（8/16 步实际计算）
- 跳过的步复用上一步的 flow prediction（cosine similarity > 0.95）
- 加速比: ~1.8x

### Phase 6: Scheduler Step

```
VideoActionScheduler.step():
  → FlowUniPC.step(video_pred, t, video_latents)  # ~0.2ms
  → FlowUniPC.step(action_pred, t, action_latents) # ~0.1ms
  → .contiguous() + cuda.synchronize()              # ~0.1ms
```

- 纯算术运算，开销极低
- CFG parallel 的 synchronize 增加微量开销

## 通信开销（CFG Parallel）

| 操作 | 数据量 | 典型耗时 (PCIe Gen5) |
|------|--------|---------------------|
| all_gather (video pred) | ~2 MB/step | ~0.1ms |
| all_gather (action pred) | ~0.8 KB/step | <0.01ms |
| cuda.synchronize | — | ~0.05ms |
| **每步通信总计** | | **~0.15ms** |
| **16 步总计** | | **~2.4ms** |

通信开销占总延迟 <1%，不是瓶颈。

## Serving 层开销

| 操作 | 典型耗时 |
|------|----------|
| WebSocket receive | ~0.1ms |
| msgpack unpack (3 images × 180×320×3) | ~2ms |
| Observation 格式转换 | ~0.5ms |
| OmniDiffusionRequest 构建 | ~0.1ms |
| asyncio.to_thread 调度 | ~0.1ms |
| msgpack pack (actions 24×8) | ~0.1ms |
| WebSocket send | ~0.1ms |
| **Serving 层总计** | **~3ms** |

Serving 层开销占总延迟 <1%，不是瓶颈。

## 性能瓶颈总结

```
┌─────────────────────────────────────────────────────┐
│              端到端延迟分布 (H100, 后续调用)            │
│                                                     │
│  ████████████████████████████████████░░░░░░ 83%  DiT│
│  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 13%  KV │
│  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2%  VAE│
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1%  Sch│
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ <1%  Net│
└─────────────────────────────────────────────────────┘

DiT forward 是绝对瓶颈 (83%)
```

## 优化路线

| 优化 | 预期加速 | 复杂度 | 状态 |
|------|---------|--------|------|
| DiT Cache（跳步） | 1.8x | 低 | DreamZero 已内置 |
| CFG Parallel (2 GPU) | ~1.8x | 中 | 已实现 |
| torch.compile | 1.2-1.5x | 低 | DreamZero 支持 |
| TensorRT engine | 2-3x | 高 | DreamZero 支持 (GB200) |
| Text/CLIP 缓存 | -50ms 首次 | 低 | 已实现 |
| FlashAttention 3 | 1.1-1.2x | 低 | DreamZero 支持 |
| FP8 量化 | 1.3-1.5x | 中 | 待实现 |
