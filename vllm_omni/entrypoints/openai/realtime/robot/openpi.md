# OpenPI Robot Serving 设计文档

## 概述

`/v1/realtime/robot/openpi` 是 vllm-omni 中用于机器人策略推理的 WebSocket 端点，协议完全兼容 DreamZero / OpenPI。

## 架构对应关系

```
DreamZero 原始                              vllm-omni
────────────────────────────────           ────────────────────────────────
policy_server.py                           openpi_connection.py
  WebsocketPolicyServer                      RobotRealtimeConnection
  ├─ _handler()                              ├─ handle_connection()
  │   ├─ send(pack(server_config))           │   ├─ send_bytes(_pack(metadata))
  │   ├─ recv → unpackb(obs)                 │   ├─ receive() → _unpack(bytes)
  │   ├─ reset → policy.reset(obs)           │   ├─ reset → serving.reset(obs)
  │   ├─ infer → policy.infer(obs)           │   ├─ infer → serving.infer(obs)
  │   └─ send(pack(action))                  │   └─ send_bytes(_pack(actions))
  │                                          │
  └─ 异常 → send(traceback)                  └─ 异常 → send_text(traceback)

socket_test_optimized_AR.py                openpi_serving.py
  ARDroidRoboarenaPolicy                     ServingRealtimeRobotOpenPI
  ├─ infer(obs) → ndarray                   ├─ infer(obs) → ndarray
  ├─ reset(obs)                              ├─ reset(obs)
  ├─ _convert_observation()                  ├─ transform.transform_input(obs)
  │   roboarena keys → AR_droid keys         │   数据集 key → 统一 key (via Transform)
  ├─ _extract_actions()                      ├─ transform.transform_output(result)
  │   dict → ndarray(N, 8)                   │   result → ndarray
  └─ 内部调 groot_policy.forward()           └─ 内部调 engine_client.step()

                                           transform/
                                             base.py: RobotPolicyTransform 基类 + TRANSFORMS 注册表
                                             droid.py: DROID 数据集 key 映射
                                             roboarena.py: RoboArena 数据集 key 映射

test_client_AR.py                          test_client_AR.py (不改代码，只改 URL)
  WebsocketClientPolicy                     WebsocketClientPolicy
  ├─ connect → recv metadata                ├─ connect → recv metadata
  ├─ infer(obs) → send/recv                 ├─ infer(obs) → send/recv
  └─ reset({}) → send/recv                  └─ reset({}) → send/recv
```

## 协议规范

序列化：`openpi_client.msgpack_numpy`（OpenPI 自有实现，`__ndarray__` 编码格式）

```
Client                                Server
  │                                      │
  │──── connect ────────────────────────>│
  │                                      │
  │<──── metadata (msgpack binary) ─────│
  │  {image_resolution, n_external_cameras,
  │   needs_wrist_camera, needs_stereo_camera,
  │   needs_session_id, action_space}
  │                                      │
  │──── obs (msgpack binary) ──────────>│
  │  {endpoint: "infer",                 │
  │   session_id: str,                   │
  │   observation/exterior_image_0_left, │
  │   observation/joint_position, ...}   │
  │                                      │
  │<──── actions (msgpack binary) ──────│
  │  ndarray (N, 8) float32              │
  │                                      │
  │──── reset (msgpack binary) ────────>│
  │  {endpoint: "reset"}                 │
  │                                      │
  │<──── "reset successful" (text) ─────│
  │                                      │
  │──── disconnect ────────────────────>│
```

## 层职责

| 层 | 文件 | 职责 | 状态 |
|---|---|---|---|
| **端点注册** | `api_server.py` | 注册 `/v1/realtime/robot/openpi`，初始化 serving | 无状态 |
| **协议层** | `openpi_connection.py` | WebSocket 收发、msgpack 编解码、错误处理 | 无状态 |
| **Transform 层** | `transform/base.py` + `droid.py` / `roboarena.py` | 数据集 key → 统一 key 映射 | 无状态 |
| **业务层** | `openpi_serving.py` | transform 路由、request 构建、engine 调用、action 提取 | session 跟踪 |
| **状态层** | `state_dreamzero.py` (pipeline 内) | 帧累积、KV cache、编码缓存 | **有状态** |
| **引擎层** | `DiffusionEngine` / `EngineClient` | 实际模型推理 | — |

## Transform 机制

Transform 是无状态的数据集适配层，按 `obs["embodiment"]` 路由：

```python
# 注册
register_transform("droid", DroidTransform())
register_transform("roboarena", RoboArenaTransform())

# 路由 (openpi_serving.py)
transform = get_transform(obs.get("embodiment", "roboarena"))
unified_obs = transform.transform_input(obs)
```

### 统一 key 格式

| 统一 key | 说明 |
|----------|------|
| `images/exterior_0` | 外部相机 0 |
| `images/exterior_1` | 外部相机 1 (可选) |
| `images/wrist` | 手腕相机 (可选) |
| `state/joint_position` | 关节位置 |
| `state/gripper_position` | 夹爪位置 |
| `prompt` | 语言指令 |

### 数据集 key 映射

| 数据集 | 原始 key | → 统一 key |
|--------|---------|-----------|
| DROID | `observation/exterior_image_1_left` | `images/exterior_0` |
| DROID | `observation/wrist_image_left` | `images/wrist` |
| RoboArena | `observation/exterior_image_0_left` | `images/exterior_0` |
| RoboArena | `observation/exterior_image_1_left` | `images/exterior_1` |
| RoboArena | `observation/wrist_image_left` | `images/wrist` |

## Engine 兼容性

`ServingRealtimeRobotOpenPI` 接受任意 `engine_client`：

| engine 类型 | 推理方式 | 适用模型 |
|---|---|---|
| `DiffusionEngine` | `engine_client.step(OmniDiffusionRequest)` | DreamZero, 扩散类策略 |
| `EngineClient` (LLM) | `engine_client.generate(...)` (需覆写 `_build_request`) | OpenVLA, RT-2 等 |

默认 `_build_request()` 构建 `OmniDiffusionRequest`。LLM 模型需子类覆写。

## 与 /v1/realtime 的关系

```
/v1/realtime                          /v1/realtime/robot/openpi
├─ OpenAIServingRealtime              ├─ ServingRealtimeRobotOpenPI
├─ RealtimeConnection                 ├─ RobotRealtimeConnection
├─ JSON text frames                   ├─ msgpack binary frames
├─ LLM EngineClient only             ├─ 任意 engine
├─ audio stream → text output         ├─ observation → action output
└─ 继承 OpenAIServing                 └─ 独立类（不继承 OpenAIServing）
```

两者是平行关系，共享 `api_server.py` 注册模式，不共享实现。

## 文件结构

```
vllm_omni/entrypoints/openai/realtime/robot/
├── openpi_connection.py       # 协议层：WebSocket + msgpack
├── openpi_serving.py          # 业务层：transform 路由 + engine 调用
├── openpi.md                  # 本文档
├── dreamzero.md               # DreamZero 推理链路设计
└── transform/
    ├── __init__.py
    ├── base.py                # RobotPolicyTransform 基类 + 注册表
    ├── droid.py               # DROID 数据集 transform
    └── roboarena.py           # RoboArena 数据集 transform

vllm_omni/diffusion/models/dreamzero/
├── state_dreamzero.py         # 模型状态：帧累积 + KV cache (统一 key)
├── pipeline_dreamzero.py      # 主 pipeline (待实现)
└── modeling/                  # 模型组件 (待实现)
```
