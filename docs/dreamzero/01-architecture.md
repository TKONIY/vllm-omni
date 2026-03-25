# 1. 架构总览：DreamZero 在 vllm-omni 中的位置

## 什么是 DreamZero

DreamZero 是一个**联合视频-动作扩散模型**（World Action Model），基于 Wan2.1-I2V-14B（140 亿参数）。它在一次前向推理中同时预测：
- **视频帧**：未来的场景画面
- **机器人动作**：7 自由度关节位置 + 1 夹爪 = 8 维动作向量 × 24 步

## 架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                        WebSocket Client                         │
│                    (test_client_AR.py / OpenPI)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ msgpack binary
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              /v1/world/roboarena (FastAPI WebSocket)             │
│                                                                 │
│  OmniWorldStreamHandler                                         │
│  ├─ 接收 observation (图像+关节+prompt)                          │
│  ├─ WorldSessionStore 管理会话状态                                │
│  ├─ 构建 OmniDiffusionRequest                                   │
│  └─ 调用 DiffusionEngine.step()                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DreamZeroPipeline.forward()                   │
│                                                                 │
│  1. _encode_prompt()     → UMT5-XXL 文本编码                    │
│  2. VAE encode           → 图像 → latent                        │
│  3. _prefill_kv_cache()  → 首帧编码进 KV cache                  │
│  4. diffuse()            → 去噪循环                              │
│     ├─ predict_noise_maybe_with_cfg()                           │
│     │   ├─ rank 0: CausalWanModel(positive_kwargs)              │
│     │   ├─ rank 1: CausalWanModel(negative_kwargs)              │
│     │   ├─ all_gather 交换结果                                   │
│     │   └─ combine_cfg_noise: 视频CFG + 动作positive-only        │
│     ├─ scheduler_step_maybe_with_cfg()                          │
│     │   └─ VideoActionScheduler.step()                          │
│     │       ├─ FlowUniPC (video)                                │
│     │       └─ FlowUniPC (action)                               │
│     └─ _synchronize_cfg_parallel_step_output()                  │
│         └─ .contiguous() + cuda.synchronize()                   │
│  5. 返回 DiffusionOutput(custom_output={"actions": ndarray})    │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
vllm_omni/diffusion/models/dreamzero/
├── __init__.py
├── pipeline_dreamzero.py              # 596 行 — Pipeline 主类
│   ├── VideoActionScheduler           # 组合调度器（~20 行）
│   └── DreamZeroPipeline              # 主 pipeline
│       ├── __init__                   # 初始化组件
│       ├── predict_noise              # 调用 CausalWanModel
│       ├── combine_cfg_noise          # 视频CFG + 动作positive
│       ├── _prefill_kv_cache          # KV cache 预填充
│       ├── diffuse                    # 去噪循环
│       ├── forward                    # 端到端推理入口
│       └── load_weights               # 权重加载
└── modeling/
    ├── __init__.py
    ├── causal_wan_model.py            # 1983 行 — 核心 DiT
    │   ├── RoPE utilities             # 旋转位置编码
    │   ├── CausalWanSelfAttention     # 因果自注意力 + KV cache
    │   ├── CausalWanAttentionBlock    # Transformer block
    │   ├── CausalHead                 # 输出投影
    │   └── CausalWanModel             # 顶层模型类
    └── action_encoder.py              # 164 行 — 动作编解码
        ├── SinusoidalPositionalEncoding
        ├── CategorySpecificLinear     # 按机器人类型选择权重
        ├── CategorySpecificMLP        # 状态编码/动作解码
        └── MultiEmbodimentActionEncoder

vllm_omni/entrypoints/openai/
├── serving_world_stream.py            # WebSocket handler
├── session_manager.py                 # 会话状态管理
├── protocol/world.py                  # 协议定义
└── api_server.py                      # 注册 /v1/world/roboarena

tests/dreamzero/                       # 6 个测试文件，22 个用例
```

## 与现有组件的复用关系

| 组件 | 来源 | 复用方式 |
|------|------|----------|
| Text Encoder (UMT5-XXL) | Wan2.2 | `from_pretrained` 直接复用 |
| VAE (Wan2.1) | Wan2.2 | `DistributedAutoencoderKLWan` 直接复用 |
| Scheduler (FlowUniPC) | 公共 | `scheduling_flow_unipc_multistep.py` 直接复用 |
| CFG Parallel | 公共 | `CFGParallelMixin` + PR #2063 tuple 支持 |
| DiT Transformer | **新写** | CausalWanModel（基于 WanTransformer3DModel 风格） |
| Action Encoder | **新写** | MultiEmbodimentActionEncoder（DreamZero 特有） |
| WebSocket Handler | **新写** | 参考 TTS streaming handler 模式 |

## 关键设计决策

### 为什么用 OpenPI 风格而不是 LeRobot gRPC？

DreamZero 原生就是 msgpack WebSocket 协议，跟 OpenPI 几乎一致。选 OpenPI 风格：
- 适配成本接近零（`test_client_AR.py` 只改 URL 即可连接）
- 简单（server ~200 行）
- 跨语言友好（msgpack 有 C++/Rust/JS 库）

### 为什么动作不做 CFG？

DreamZero 的 CFG 只应用于视频预测。动作预测直接使用条件（positive）分支：
```python
def combine_cfg_noise(self, pos, neg, scale, normalize):
    (video_pos, action_pos) = pos
    (video_neg, _) = neg
    video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
    return (video_combined, action_pos)  # 动作：仅取 positive
```
原因：动作应直接跟随语言指令，无条件分支的"漫无目的"动作没有意义。

### 为什么需要 KV Cache？

DreamZero 是**自回归**推理：每次调用处理一个帧块（1帧），KV cache 累积历史上下文。
- 首次调用：编码首帧 → KV cache 从 0 开始
- 后续调用：只处理新帧 → KV cache 追加
- Reset：清空 KV cache → 开始新 episode
