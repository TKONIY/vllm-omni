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
  ├─ _convert_observation()                  ├─ _build_request()
  │   roboarena keys → AR_droid keys         │   OpenPI keys → OmniDiffusionRequest
  ├─ _extract_actions()                      ├─ _extract_actions()
  │   dict → ndarray(N, 8)                   │   DiffusionOutput → ndarray
  └─ 内部调 groot_policy.forward()           └─ 内部调 engine_client.step()

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

| 层 | 文件 | 职责 |
|---|---|---|
| **端点注册** | `api_server.py` | 注册 `/v1/realtime/robot/openpi`，初始化 serving |
| **协议层** | `openpi_connection.py` | WebSocket 收发、msgpack 编解码、错误处理 |
| **业务层** | `openpi_serving.py` | obs→request 转换、调 engine、result→actions 提取 |
| **引擎层** | `DiffusionEngine` / `EngineClient` | 实际模型推理 |

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
