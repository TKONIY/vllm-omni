# 5. WebSocket Serving：OpenPI 风格协议与会话管理

## 协议设计

DreamZero 的 WebSocket 协议完全兼容 OpenPI。`test_client_AR.py` 只需改 URL 即可连接 vllm-omni。

### 协议流程

```
Client                                    Server
  │                                          │
  │──── WebSocket connect ──────────────────>│
  │                                          │
  │<──── metadata (msgpack) ────────────────│
  │  {image_resolution, n_external_cameras,  │
  │   needs_wrist_camera, action_space, ...} │
  │                                          │
  │──── obs (msgpack) ─────────────────────>│
  │  {endpoint: "infer",                     │
  │   session_id, observation/*, prompt}     │
  │                                          │
  │<──── actions (msgpack ndarray) ─────────│
  │  np.ndarray (24, 8) float32              │
  │                                          │
  │──── reset (msgpack) ───────────────────>│
  │  {endpoint: "reset"}                     │
  │                                          │
  │<──── "reset successful" (plain text) ───│
  │                                          │
  │──── disconnect ─────────────────────────>│
```

### 三个关键协议点

| 消息类型 | 格式 | 说明 |
|---------|------|------|
| **metadata** | `msgpack(dict)` | 字段必须兼容 `PolicyServerConfig(**metadata)` |
| **infer response** | `msgpack(ndarray)` | 直接发 raw ndarray，**不是** dict |
| **reset response** | plain text string | `"reset successful"`，**不是** msgpack |

这三个点是我们在精度对齐过程中发现并修复的，确保与 DreamZero 原始 server 行为一致。

## Metadata 字段

```python
metadata = {
    "image_resolution": (180, 320),    # (H, W)
    "n_external_cameras": 2,           # 外部相机数量
    "needs_wrist_camera": True,        # 需要手腕相机
    "needs_stereo_camera": False,      # 不需要立体视觉
    "needs_session_id": True,          # 需要 session ID
    "action_space": "joint_position",  # 关节位置空间
}
```

## Observation Key 映射

```
客户端 (OpenPI 风格)              →  DreamZero 内部
observation/exterior_image_0_left →  video.exterior_image_1_left
observation/exterior_image_1_left →  video.exterior_image_2_left
observation/wrist_image_left      →  video.wrist_image_left
observation/joint_position        →  state.joint_position
observation/gripper_position      →  state.gripper_position
prompt                            →  annotation.language.action_text
```

## 会话管理

### WorldSessionState

```python
@dataclass
class WorldSessionState:
    session_id: str
    call_count: int = 0
    created_at: float
    last_active_at: float
    needs_reset: bool = True  # 首次调用自动 reset KV cache

class DreamZeroSessionState(WorldSessionState):
    exterior_image_1_left: list[np.ndarray]  # 帧缓冲
    exterior_image_2_left: list[np.ndarray]
    wrist_image_left: list[np.ndarray]
```

### 会话生命周期

```
新 session_id    → 创建 session, needs_reset=True → KV cache 初始化
同 session_id    → 复用 session, needs_reset=False → KV cache 继续累积
endpoint="reset" → 重置 session, needs_reset=True → KV cache 清空
WebSocket 断开   → 销毁 session
TTL 超时 (300s)  → 自动过期清理
```

### WorldSessionStore

线程安全（`threading.Lock`），支持：
- `get_or_create(session_id)` — 获取或创建
- `reset(session_id)` — 重置帧缓冲 + 标记需要 KV cache reset
- `destroy(session_id)` — 销毁
- `cleanup_expired(ttl_seconds)` — 清理超时会话

## 文件结构

```
vllm_omni/entrypoints/openai/
├── api_server.py              # @router.websocket("/v1/world/roboarena")
├── serving_world_stream.py    # OmniWorldStreamHandler (~170 行)
│   ├── handle_session()       # 主循环：metadata → infer/reset loop
│   ├── _build_request()       # obs dict → OmniDiffusionRequest
│   └── _extract_actions()     # DiffusionOutput → ndarray
├── session_manager.py         # WorldSessionState/Store (~100 行)
└── protocol/world.py          # WorldModelMetadata pydantic
```

## 与 TTS Streaming 的对比

| 维度 | TTS Streaming | DreamZero World |
|------|--------------|-----------------|
| 协议 | JSON over WebSocket | **msgpack over WebSocket** |
| 状态 | 低（文本缓冲） | **高**（KV cache + 帧缓冲） |
| 并发 | 多会话 | **单 GPU 单会话** |
| 消息方向 | 文本输入 → 音频输出 | 观测输入 → 动作输出 |
| 编码 | 文本分句 | **图像+关节 msgpack** |
