# 2. 端到端执行流程

## 一次推理调用的完整路径

以 `test_client_AR.py` 发送一帧观测为例，追踪数据从客户端到 action 输出的完整路径。

### Step 0: 客户端连接

```python
# test_client_AR.py
client = WebsocketClientPolicy(host="localhost", port=8000)
# → WebSocket 连接 ws://localhost:8000/v1/world/roboarena
# ← 服务端发送 metadata (msgpack):
#    {"image_resolution": (180, 320), "n_external_cameras": 2,
#     "needs_wrist_camera": True, "action_space": "joint_position", ...}
```

### Step 1: 发送观测

```python
obs = {
    "endpoint": "infer",
    "session_id": "episode-001",
    "observation/exterior_image_0_left": np.ndarray(180, 320, 3),  # uint8
    "observation/exterior_image_1_left": np.ndarray(180, 320, 3),
    "observation/wrist_image_left": np.ndarray(180, 320, 3),
    "observation/joint_position": np.ndarray(7,),   # float32
    "observation/gripper_position": np.ndarray(1,),
    "prompt": "pick up the red block",
}
actions = client.infer(obs)  # → msgpack pack → WebSocket send
```

### Step 2: Handler 处理 (`serving_world_stream.py`)

```
OmniWorldStreamHandler.handle_session():
  1. msgpack_numpy.unpackb(raw) → obs dict
  2. 提取 endpoint="infer", session_id="episode-001"
  3. 删除 obs["endpoint"]（DreamZero 约定）
  4. WorldSessionStore.get_or_create("episode-001")
     → 首次调用：创建 DreamZeroSessionState(needs_reset=True)
  5. _build_request(obs, session):
     → 构造 OmniDiffusionRequest:
       prompts=["pick up the red block"]
       extra_args={
         "reset": True,      # 首次调用需要 reset KV cache
         "session_id": "episode-001",
         "images": {...},    # 3 个相机图像
         "state": {...},     # 关节 + 夹爪
       }
  6. engine.step(request)  # 在线程池中同步执行
```

### Step 3: Pipeline 推理 (`pipeline_dreamzero.py`)

```
DreamZeroPipeline.forward(req):
  1. extra_args["reset"]=True → _reset_session()
     → 清空 kv_cache, clip_feas, ys, current_start_frame=0

  2. _encode_prompt("pick up the red block")
     → tokenizer → UMT5-XXL → prompt_embeds [1, 512, 4096]
     → 空 prompt → negative_prompt_embeds（CFG uncond 分支）

  3. VAE encode: observation images → video_latents [1, 16, 1, H/8, W/8]

  4. _prefill_kv_cache():
     → current_start_frame == 0:
       → _create_kv_caches(): 40 层 × [2, 1, 0, 40, 128] (空)
       → _create_crossattn_caches(): 40 层 × [2, 1, 512, 40, 128]
       → CausalWanModel(first_frame, timestep=0, action=None)
         → 编码首帧进 KV cache（side effect）
       → CausalWanModel(first_frame, timestep=0, negative_prompt)
         → 编码首帧进 kv_cache_neg
       → current_start_frame = 1

  5. 准备噪声:
     noise_video = randn_like(video_latents)
     noise_action = randn(1, 24, 8)  # 24 步 × 8 维动作

  6. 创建调度器:
     video_scheduler = FlowUniPCMultistepScheduler(shift=5.0)
     action_scheduler = FlowUniPCMultistepScheduler(shift=5.0)
     video_action_scheduler = VideoActionScheduler(video, action)

  7. diffuse() 去噪循环 (16 步):
     for t in timesteps:
       → predict_noise_maybe_with_cfg()  [见 Step 4]
       → scheduler_step_maybe_with_cfg()
         → VideoActionScheduler.step()
           → video_scheduler.step(video_pred, t, video_latents)
           → action_scheduler.step(action_pred, t, action_latents)
       → _synchronize_cfg_parallel_step_output()
         → .contiguous() + cuda.synchronize()

  8. current_start_frame += 1

  9. return DiffusionOutput(
       output=video_latents,
       custom_output={"actions": action_latents.numpy()}  # (1, 24, 8)
     )
```

### Step 4: CFG Parallel 细节 (`cfg_parallel.py`)

```
predict_noise_maybe_with_cfg():
  cfg_world_size > 1 时 (CFG Parallel):
    rank 0: video_pred, action_pred = CausalWanModel(positive_kwargs)
    rank 1: video_pred, action_pred = CausalWanModel(negative_kwargs)
    
    all_gather 交换:
      gathered_video = [rank0_video, rank1_video]
      gathered_action = [rank0_action, rank1_action]
    
    所有 rank 本地 combine:
      video_combined = neg + 5.0 * (pos - neg)  # 标准 CFG
      action_combined = action_pos               # 仅取 positive
    
    返回 (video_combined, action_combined) — 所有 rank 结果一致

  cfg_world_size == 1 时 (顺序):
    pos_video, pos_action = CausalWanModel(positive_kwargs)
    neg_video, neg_action = CausalWanModel(negative_kwargs)
    combine 同上
```

### Step 5: 返回结果

```
Handler:
  actions = _extract_actions(result)  # np.ndarray (24, 8)
  await websocket.send_bytes(msgpack.pack(actions))  # 直接发 ndarray

Client:
  actions = msgpack_numpy.unpackb(response)  # np.ndarray (24, 8)
  # 执行前 8 个动作，然后发送下一帧观测
```

## 多轮 AR 推理的状态流转

```
Call 1 (reset):
  current_start_frame: 0 → 1
  KV cache: 空 → [首帧编码]
  输出: actions[0:24]

Call 2 (infer):
  current_start_frame: 1 → 2
  KV cache: [首帧] → [首帧, 第二帧]
  输出: actions[24:48]

Call 3 (infer):
  current_start_frame: 2 → 3
  KV cache: [首帧, 第二帧] → [首帧, 第二帧, 第三帧]
  输出: actions[48:72]

...直到 reset 或 KV cache 超过 local_attn_size 自动 reset
```

## 数据形状速查

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入图像 | observation/image | `(H, W, 3)` uint8 | 180×320 RGB |
| VAE 输入 | video tensor | `(1, 3, T, H, W)` float | [-1, 1] 归一化 |
| VAE latent | video_latents | `(1, 16, T, H/8, W/8)` bf16 | 16 通道 latent |
| Patch 后 | hidden_states | `(1, seq_len, 5120)` | seq_len = T×(H/16)×(W/16) |
| 动作噪声 | noise_action | `(1, 24, 8)` bf16 | 24 步 × 8 维 |
| 文本编码 | prompt_embeds | `(1, 512, 4096)` | UMT5-XXL 输出 |
| KV Cache | per-layer | `(2, B, seq, 40, 128)` | [K,V] × B × seq × heads × d |
| 输出动作 | actions | `(24, 8)` float32 | 7 joints + 1 gripper |
