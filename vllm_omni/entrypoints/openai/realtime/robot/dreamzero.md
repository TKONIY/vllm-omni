# DreamZero 推理调用链梳理

## 原始 DreamZero 调用链

```
test_client_AR.py
  │ obs dict (OpenPI format: observation/xxx keys)
  ▼
socket_test_optimized_AR.py: ARDroidRoboarenaPolicy.infer(obs)
  │ 1. _convert_observation(obs)     ← key映射 + 帧累积 + reshape
  │ 2. dist.broadcast(signal)        ← 分布式通信
  │ 3. Batch(obs=converted_obs)      ← 包成 tianshou.Batch
  ▼
sim_policy.py: GrootSimPolicy.lazy_joint_forward_causal(batch)
  │ 1. unsqueeze_dict_values         ← 加 batch dim
  │ 2. apply(batch)                  ← 数据归一化 (action normalize等)
  │ 3. bf16 转换
  ▼
base_vla.py: VLA.lazy_joint_video_action_causal(normalized_input)
  │ 1. prepare_input(inputs)         ← 拆成 backbone_inputs + action_inputs
  │ 2. backbone(backbone_inputs)     ← backbone 是 Identity (不做事)
  ▼
wan_flow_matching_action_tf.py: WANPolicyHead.lazy_joint_video_action(backbone_out, action_input)
  │ 1. 视频预处理 (uint8→float, normalize, reshape)
  │ 2. text encoder (UMT5-XXL)
  │ 3. image encoder (CLIP) + VAE encode → clip_feas, ys
  │ 4. KV cache 创建/prefill
  │ 5. 去噪循环 (_run_diffusion_steps × N)
  │ 6. scheduler.step (video + action)
  ▼
wan_video_dit_action_casual_chunk.py: CausalWanModel.forward(...)
  │ (实际 DiT transformer forward)
  ▼
返回: BatchFeature(action_pred=..., video_pred=...)
  ▼
sim_policy.py: unapply(normalized_action)  ← 动作反归一化
  ▼
socket_test_optimized_AR.py: _convert_action(action_dict)  ← dict → ndarray(N,8)
  ▼
client 收到 ndarray(N, 8)
```

## vllm-omni 需要 vs 不需要

| 层 | DreamZero 原始 | vllm-omni 需要？ | 原因 |
|---|---|---|---|
| `tianshou.Batch` | 包装 obs | **不需要** | 纯 dict 直接传 |
| `GrootSimPolicy` | 模型加载+归一化+反归一化 | **部分需要** | 归一化/反归一化逻辑需要，但不要 tianshou/hydra 框架 |
| `base_vla.VLA` | prepare_input + backbone | **不需要** | backbone 是 Identity，prepare_input 只是拆 dict |
| `ActionHead` 基类 | 抽象接口 | **不需要** | 直接实现 |
| `dist.broadcast/barrier` | 分布式通信 | **不需要** | CFGParallelMixin 管 |
| `_convert_observation` | key 映射+帧累积 | **需要** | pipeline 内部做（state_dreamzero.py） |
| `apply/unapply` | 动作归一化/反归一化 | **需要** | pipeline 层 |

## vllm-omni 目标调用链

```
test_client_AR.py (不改代码，只改 URL)
  │ obs dict (数据集格式: observation/exterior_image_0_left 等)
  ▼
openpi_connection.py: RobotRealtimeConnection.handle_connection()
  │ 1. _unpack(bytes) → obs dict
  │ 2. obs.pop("endpoint")
  ▼
openpi_serving.py: ServingRealtimeRobotOpenPI.infer(obs)
  │ 1. session 跟踪
  │ 2. transform = get_transform(obs["embodiment"] or default)
  │ 3. unified_obs = transform.transform_input(obs)
  │    数据集 key → 统一 key (images/exterior_0, state/joint_position 等)
  │ 4. _build_request(unified_obs) → OmniDiffusionRequest
  │    unified_obs 透传到 extra_args["unified_obs"]
  ▼
DiffusionEngine.step(request)
  ▼
pipeline_dreamzero.py: DreamZeroPipeline.forward(req)
  │
  │ ┌─── state_dreamzero.py: DreamZeroState ──────────────┐
  │ │ 持久状态（跨 forward 调用存活，模型相关）:              │
  │ │ - frame_buffers: 帧累积 buffer（按统一 key）           │
  │ │ - kv_cache / kv_cache_neg: KV cache (40层)            │
  │ │ - crossattn_cache / crossattn_cache_neg               │
  │ │ - clip_feas / ys: 编码缓存                             │
  │ │ - current_start_frame: AR 帧计数器                     │
  │ │                                                        │
  │ │ 方法:                                                   │
  │ │ - accumulate_frames(unified_obs) → 帧累积              │
  │ │ - reset() → 清空所有状态                                │
  │ │ - should_reset(...) → 判断是否需要 reset                │
  │ │ - create/update/get kv_caches                          │
  │ └────────────────────────────────────────────────────────┘
  │
  │ 1. state.accumulate_frames(unified_obs) → 多帧视频
  │ 2. 视频预处理 (uint8→float, normalize)
  │ 3. text encoder (复用 UMT5)
  │ 4. image encoder (CLIP) + VAE encode
  │ 5. state 管理 KV cache 创建/prefill
  │ 6. diffuse() 去噪循环
  │    ├─ predict_noise_maybe_with_cfg (CFGParallelMixin)
  │    ├─ scheduler_step_maybe_with_cfg (VideoActionScheduler)
  │    └─ _synchronize_cfg_parallel_step_output
  │ 7. 动作反归一化
  ▼
返回: DiffusionOutput(custom_output={"actions": ndarray(N,8)})
  ▼
openpi_serving.py: transform.transform_output(result) → ndarray
  ▼
openpi_connection.py: _pack(actions) → send_bytes
  ▼
client 收到 ndarray(N, 8)
```

## 三层分工

```
Transform (数据集相关, 无状态)          State (模型相关, 有状态)          Pipeline (推理)
┌──────────────────────┐          ┌────────────────────────┐    ┌──────────────┐
│ DroidTransform       │          │ DreamZeroState         │    │ DreamZero    │
│ RoboArenaTransform   │──key映射──▶│ accumulate_frames()   │──▶│ Pipeline     │
│ LiberoTransform      │          │ KV cache management   │    │ .forward()   │
│ ...                  │          │ encoding cache        │    │              │
└──────────────────────┘          └────────────────────────┘    └──────────────┘
  observation/xxx                   images/exterior_0              model input
  → images/exterior_0               → (T,H,W,3) stacked
```

## Transform 注册机制

```python
# transform/base.py — 基类 + 注册表
TRANSFORMS: dict[str, RobotPolicyTransform] = {}
register_transform("droid", DroidTransform())
register_transform("roboarena", RoboArenaTransform())

# openpi_serving.py — 按 obs["embodiment"] 路由
transform = get_transform(obs.get("embodiment", "roboarena"))
unified_obs = transform.transform_input(obs)
```

## 统一 key 格式

Transform 输出 / State 消费的统一格式：

| 统一 key | 类型 | 说明 |
|----------|------|------|
| `images/exterior_0` | ndarray (H,W,3) 或 (T,H,W,3) | 外部相机 0 |
| `images/exterior_1` | ndarray (H,W,3) 或 (T,H,W,3) | 外部相机 1 (可选) |
| `images/wrist` | ndarray (H,W,3) 或 (T,H,W,3) | 手腕相机 (可选) |
| `state/joint_position` | ndarray (N,) | 关节位置 |
| `state/gripper_position` | ndarray (N,) | 夹爪位置 |
| `prompt` | str | 语言指令 |
| `session_id` | str | 会话 ID (透传) |

## 需要实现的文件

### 核心文件（需新写）

| 文件 | 对应 DreamZero 原始 | 说明 |
|------|-------------------|------|
| `pipeline_dreamzero.py` | `WANPolicyHead.lazy_joint_video_action` | 主 pipeline，去掉 tianshou/BatchFeature 包装 |
| `state_dreamzero.py` | `ARDroidRoboarenaPolicy._frame_buffers` + `WANPolicyHead.kv_cache*` | 跨调用持久状态：帧累积 + KV cache + CLIP/VAE 缓存 |
| `modeling/causal_wan_model.py` | `CausalWanModel` | 40层 DiT transformer |
| `modeling/wan_video_image_encoder.py` | `WanImageEncoder` | CLIP ViT-H/14 编码器 |

### state_dreamzero.py 职责

```python
class DreamZeroState:
    """Pipeline 内部持久状态，跨 forward() 调用存活。

    原始 DreamZero 中这些状态分散在三个地方：
    - ARDroidRoboarenaPolicy._frame_buffers   → 帧累积
    - WANPolicyHead.kv_cache1/kv_cache_neg    → KV cache
    - WANPolicyHead.clip_feas/ys              → 编码缓存
    
    合并到一个文件，统一生命周期管理。
    """
    # 帧累积 (从 ARDroidRoboarenaPolicy 搬来)
    frame_buffers: dict[str, list[ndarray]]
    call_count: int
    
    # KV cache (从 WANPolicyHead 搬来)
    kv_cache: list[Tensor] | None
    kv_cache_neg: list[Tensor] | None
    crossattn_cache: list[Tensor] | None
    crossattn_cache_neg: list[Tensor] | None
    current_start_frame: int
    
    # 编码缓存 (从 WANPolicyHead 搬来)
    clip_feas: Tensor | None
    ys: Tensor | None
    language: Tensor | None  # 用于检测 prompt 变化
    
    def accumulate_frames(self, openpi_obs) → dict[str, ndarray]:
        """帧累积，返回 (T,H,W,3) per camera key"""
    
    def reset(self):
        """清空所有状态"""
    
    def should_reset(self, text_tokens, num_video_frames, local_attn_size) → bool:
        """判断是否需要 reset"""
    
    def create_kv_caches(self, batch_size, dtype, device, num_layers, num_heads, head_dim):
        """初始化空 KV cache + crossattn cache"""
    
    def update_kv_cache(self, layer_index, updated_kv, is_negative=False):
        """更新单层 KV cache"""
    
    def get_kv_caches(self, is_negative=False) → list[Tensor]:
        """取 cond/uncond 分支的 KV cache"""
    
    def get_crossattn_caches(self, is_negative=False) → list[Tensor]:
        """取 crossattn cache"""
```

### state_dreamzero.py 代码对应关系

| state_dreamzero.py 方法 | 原始文件 | 原始行号 | 做什么 |
|---|---|---|---|
| `accumulate_frames` L62-90 | `socket_test_optimized_AR.py` L110-144 | `_convert_observation` 的帧 buffer 部分 | 按 key append 帧 → 取最后 N 帧 → stack (T,H,W,3) |
| `reset` L98-117 | `socket_test_optimized_AR.py` L302-330 + `wan_flow_matching_action_tf.py` L185-199 | 清空帧 buffer + KV cache + 编码缓存 |
| `should_reset` L119-152 | `wan_flow_matching_action_tf.py` L968-981 | 4 个条件：language=None / language 变化 / 单帧 / overflow |
| `create_kv_caches` L160-186 | `wan_flow_matching_action_tf.py` L480-512 | KV `[2,B,0,heads,d]` + crossattn `[2,B,512,heads,d]` |
| `update_kv_cache` L188-199 | `wan_flow_matching_action_tf.py` L856-858 | `kv_cache[layer] = updated.clone()` |
| `get_kv_caches` L201-209 | `wan_flow_matching_action_tf.py` L776-791 | 返回 cond/uncond cache（去掉 ip_rank 分发，CFGParallelMixin 管） |
| `get_crossattn_caches` L211-217 | 同上 | 同上 crossattn 版本 |

### 去掉的逻辑

| 原始逻辑 | 原始位置 | 为什么去掉 |
|---|---|---|
| key remapping (`observation/xxx` → `video.xxx`) | `_convert_observation` L104-108 | pipeline 直接读 OpenPI key，不需要改名 |
| `ip_rank`/`ip_size` cache 分发 | `_get_caches` L779-785 | CFGParallelMixin 管 rank 分发 |
| `_prepare_text_inputs` rank 分发 | L793-805 | 同上 |
| state reshape `(7,)→(1,7)` | `_convert_observation` L148-163 | pipeline 自己做 |

### 可复用 vllm-omni 现有组件

| 组件 | vllm-omni 现有 | 对应 DreamZero |
|------|--------------|---------------|
| Text Encoder | `UMT5EncoderModel` | `WanTextEncoder` |
| VAE | `DistributedAutoencoderKLWan` | `WanVideoVAE` |
| Scheduler | `FlowUniPCMultistepScheduler` | `FlowUniPCMultistepScheduler` |
| CFG Parallel | `CFGParallelMixin` | 手写 `dist.broadcast/isend/irecv` |

### 从 DreamZero 提取+精简

| 文件 | 对应 DreamZero 原始 | 说明 |
|------|-------------------|------|
| `modeling/action_encoder.py` | `MultiEmbodimentActionEncoder` 等 | 动作编解码 |
| `action_transform.py` | `GrootSimPolicy.apply/unapply` | 动作归一化/反归一化 |

## 关键精简点

1. **去掉 tianshou.Batch** — 纯 dict 传递，不需要 Batch 包装
2. **去掉 VLA/backbone** — backbone 是 Identity，直接跳过
3. **去掉 hydra/OmegaConf** — 不用 hydra 实例化，直接构造
4. **去掉 dist.broadcast/barrier** — CFGParallelMixin 的 all_gather 替代
5. **去掉 GrootSimPolicy** — 把 apply/unapply 逻辑搬到 pipeline
6. **去掉 ActionHead 基类** — 不需要抽象继承
7. **状态统一到 state_dreamzero.py** — 帧累积 + KV cache + 编码缓存，一个文件管全部跨调用状态
8. **去掉 key remapping** — pipeline 直接用 OpenPI 原始 key
