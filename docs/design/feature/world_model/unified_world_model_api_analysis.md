# Unified World Model Server API Analysis

Comparison of DreamZero, OpenPI, and LeRobot serving APIs to inform vllm-omni's unified world model server design.

---

## Three Frameworks at a Glance

| | DreamZero | OpenPI (Physical Intelligence) | LeRobot (Hugging Face) |
|---|---|---|---|
| **Transport** | WebSocket (msgpack) | WebSocket (msgpack) | gRPC (protobuf + pickle) |
| **State** | Stateful (KV cache, frame buffers) | Stateless (each infer independent) | Stateful (action queue, observation de-dup) |
| **Model type** | World model (DiT, AR video+action) | VLA (pi0, pi0-FAST) | Any policy (ACT, Diffusion, pi0, VQ-BET, ...) |
| **Session** | `session_id` in obs, explicit reset | None — no session concept | Episode-based (`reset` on episode start) |
| **Action chunking** | Server returns 24 actions, client executes 8 | Server returns full horizon, client chunks via `ActionChunkBroker` | Server returns chunk, client aggregates via weighted average |
| **Multi-robot** | DROID only (hardcoded cameras/action space) | ALOHA, DROID, LIBERO, etc. (via transform chains) | Any robot (via configurable features) |
| **Image format** | `(T, H, W, 3)` uint8, 180x320 | `(H, W, 3)` uint8/float32, 224x224 | `(H, W, 3)` uint8, varies |
| **Metadata** | `PolicyServerConfig` on connect | Custom dict on connect | `PolicySetup` via gRPC |
| **Health check** | None | `GET /healthz` | gRPC `Ready()` |

---

## Protocol Comparison

### Connection Lifecycle

**DreamZero:**
```
connect → server sends PolicyServerConfig (msgpack)
loop: client sends obs (msgpack) → server returns action (msgpack)
client sends reset → server clears state
disconnect
```

**OpenPI:**
```
connect → server sends metadata dict (msgpack)
loop: client sends obs (msgpack) → server returns action dict (msgpack)
disconnect (no explicit reset/session)
```

**LeRobot:**
```
connect → client calls Ready() (gRPC)
client calls SendPolicyInstructions(policy_type, path, device)
loop: client streams SendObservations() → client calls GetActions()
disconnect
```

### Observation Format

**DreamZero (roboarena):**
```python
{
    "observation/exterior_image_0_left": np.ndarray (H, W, 3),  # or (T, H, W, 3)
    "observation/exterior_image_1_left": np.ndarray (H, W, 3),
    "observation/wrist_image_left": np.ndarray (H, W, 3),
    "observation/joint_position": np.ndarray (7,),
    "observation/gripper_position": np.ndarray (1,),
    "prompt": str,
    "session_id": str,
    "endpoint": "infer" | "reset",
}
```

**OpenPI:**
```python
{
    "observation/exterior_image_1_left": np.ndarray (224, 224, 3),  # DROID
    "observation/wrist_image_left": np.ndarray (224, 224, 3),
    "observation/joint_position": np.ndarray (7,),
    "observation/gripper_position": np.ndarray (1,),
    "prompt": str,
}
```

**LeRobot:**
```python
{
    "shoulder_pan.pos": float,       # Per-joint keys
    "shoulder_lift.pos": float,
    "gripper.pos": float,
    "side": np.ndarray (480, 640, 3),  # Camera name as key
    "up": np.ndarray (480, 640, 3),
    "task": str,
}
```

### Action Format

**DreamZero:** `np.ndarray (N, 8)` — 7 joints + 1 gripper concatenated

**OpenPI:** `dict {"actions": np.ndarray (horizon, action_dim), "state": np.ndarray, "policy_timing": {...}}`

**LeRobot:** `list[TimedAction]` — each with timestamp, action tensor, timestep

---

## Key Design Differences

### 1. Stateful vs Stateless

| Framework | Server State | Why |
|-----------|-------------|-----|
| DreamZero | Heavy: KV cache (~2GB GPU), frame buffers, CLIP cache, VAE cache | World model needs temporal context (AR inference) |
| OpenPI | None: each infer() is independent | VLA policies encode everything in the current observation |
| LeRobot | Light: action queue, last observation | Action aggregation across chunks |

**Insight for vllm-omni:** The unified API must support both stateful (DreamZero) and stateless (OpenPI) models. Session management should be **opt-in** — stateless models ignore session_id.

### 2. Transform Pipeline

| Framework | Where transforms happen |
|-----------|------------------------|
| DreamZero | Server-side: `_convert_observation` (frame accumulation), `apply` (normalize), model internal (encode) |
| OpenPI | Server-side: `InjectDefaultPrompt → Normalize → ResizeImages → TokenizePrompt → Observation.from_dict()` — chained transforms |
| LeRobot | Server-side: `preprocessor` pipeline (normalize, resize, device move), `postprocessor` (unnormalize, cpu) |

**Insight for vllm-omni:** All three do transforms server-side. OpenPI's chained transform approach is the most extensible — each robot type just adds its transform chain. This maps well to vllm-omni's `pre_process_func`.

### 3. Multi-Robot Support

| Framework | How |
|-----------|-----|
| DreamZero | Hardcoded DROID config in server |
| OpenPI | Transform chains per robot: `AlohaInputs`, `DroidInputs`, `LiberoInputs` + universal `Observation` format |
| LeRobot | Configurable `input_features`/`output_features` in policy config + robot-specific `RobotConfig` |

**Insight for vllm-omni:** OpenPI's pattern is best — server code is robot-agnostic, transforms handle the mapping. This aligns with our "handler decomposition" future work (separate protocol / embodiment / model).

### 4. Action Chunking Pattern

All three return multi-step action chunks from the server. Client-side buffering differs:

| Framework | Server returns | Client executes | Re-query trigger |
|-----------|---------------|-----------------|-----------------|
| DreamZero | 24 actions | 8 (open_loop_horizon) | Buffer exhausted |
| OpenPI | Full horizon (e.g., 10) | 1 per step via `ActionChunkBroker` | Chunk exhausted |
| LeRobot | Configurable chunk | 1 per step, with weighted aggregation | `chunk_size_threshold` (e.g., 50% remaining) |

**Insight for vllm-omni:** Server should return the full action chunk. How the client consumes it (open-loop, single-step, aggregated) is client-side policy.

---

## Unified API Proposal for vllm-omni

### Recommended: WebSocket + msgpack (OpenPI-compatible)

**Why WebSocket + msgpack:**
- DreamZero already uses it (via roboarena)
- OpenPI already uses it (same `msgpack_numpy` library)
- Lower latency than gRPC for small payloads
- Simpler than gRPC (no proto compilation, no code generation)
- LeRobot is the outlier (gRPC), but they could add a WebSocket adapter

**Why not gRPC:**
- LeRobot's gRPC adds complexity (proto compilation, streaming semantics)
- For single-request-response (DreamZero's pattern), gRPC streaming is overhead
- Most robot eval frameworks (roboarena, openpi-client) already use WebSocket

### Protocol Design

```
Client → Server:
  connect
  ← server sends metadata (msgpack dict)

  → {"type": "infer", "obs": {...}, "session_id": "...", "config": {...}}
  ← {"type": "action", "actions": ndarray, "timing": {...}}

  → {"type": "reset", "session_id": "..."}
  ← {"type": "reset.done"}

  → {"type": "configure", "transforms": [...]}    # optional: hot-reload transforms
  ← {"type": "configured"}
```

### Observation — Unified Schema

Use **OpenPI's key convention** (most widely adopted, roboarena-compatible):

```python
{
    # Images: "observation/{camera_name}" → ndarray (H, W, 3) or (T, H, W, 3)
    "observation/exterior_image_0_left": ndarray,
    "observation/wrist_image_left": ndarray,

    # State: "observation/{state_key}" → ndarray
    "observation/joint_position": ndarray (N,),
    "observation/gripper_position": ndarray (M,),

    # Language
    "prompt": str,

    # Session (optional, ignored by stateless models)
    "session_id": str | None,
}
```

**Why this format:**
- DreamZero already uses `observation/xxx` keys (roboarena)
- OpenPI uses the same keys for DROID
- LeRobot's per-joint keys (`shoulder_pan.pos`) can be flattened to `observation/joint_position` by a transform

### Action — Unified Schema

```python
{
    # Primary output: action array
    "actions": ndarray (N, action_dim),

    # Optional timing metadata
    "timing": {
        "inference_ms": float,
        "preprocess_ms": float,
    },

    # Optional session state (for stateful models)
    "session_id": str | None,
    "chunk_index": int | None,
}
```

### Server Metadata

Merge DreamZero's `PolicyServerConfig` with OpenPI's metadata dict:

```python
{
    # From DreamZero (roboarena)
    "image_resolution": [180, 320],
    "n_external_cameras": 2,
    "needs_wrist_camera": true,
    "action_space": "joint_position",
    "needs_session_id": true,

    # From OpenPI
    "model_name": "dreamzero",
    "action_dim": 8,
    "action_horizon": 24,

    # vllm-omni additions
    "stateful": true,
    "supported_embodiments": ["droid", "agibot"],
}
```

### Compatibility Layers

```
test_client_AR.py (DreamZero)     → /v1/world/roboarena  → RoboarenaProtocolAdapter
openpi WebsocketClientPolicy      → /v1/world/openpi     → OpenPIProtocolAdapter (same msgpack!)
lerobot RobotClient               → /v1/world/grpc       → gRPCAdapter (future)
new clients                       → /v1/world/stream      → NativeProtocolAdapter (JSON)
                                         |
                                    All adapters produce
                                    unified OmniDiffusionRequest
                                         |
                                    DiffusionEngine.step()
```

**Key insight:** DreamZero's roboarena protocol and OpenPI's protocol are **almost identical** — both are msgpack over WebSocket with the same observation key convention. The only difference is:
- DreamZero adds `session_id` and `endpoint` fields
- OpenPI adds timing info in response
- DreamZero returns raw numpy array, OpenPI returns dict

A single WebSocket handler can serve both by detecting which fields are present.

---

## Recommendation

**MVP:** Single `/v1/world/roboarena` endpoint (already decided) — serves DreamZero and is OpenPI-compatible with minimal adaptation.

**P1:** Add `/v1/world/stream` with JSON protocol. Add OpenPI compatibility (detect OpenPI client by absence of `endpoint` field).

**P2:** Add gRPC adapter for LeRobot compatibility. Or contribute a WebSocket client to LeRobot upstream.

**Long-term:** The unified API is essentially **OpenPI's protocol + DreamZero's session management + vllm-omni's engine integration**. The observation/action schemas are already compatible across DreamZero and OpenPI.
