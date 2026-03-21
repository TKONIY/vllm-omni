# DreamZero Server Implementation in vllm-omni (PR7: OpenPI WebSocket)

Detailed design for the OpenPI-compatible WebSocket serving layer. Serial request-response loop matching DreamZero and OpenPI's original behavior. Compatible with `test_client_AR.py`, OpenPI `WebsocketClientPolicy`, and `DreamZeroJointPosClient`.

For LeRobot gRPC serving, see `PR4_lerobot_grpc_api.md`. For async pipeline, see `PR5_async_pipeline.md`.

---

## File Structure

```
vllm_omni/entrypoints/openai/
├── api_server.py                → @router.websocket("/v1/world/openpi")
│                                  → handler.handle_session(websocket)
│
├── serving_world_stream.py      → OmniWorldStreamHandler
│                                  - WebSocket handler (MVP: roboarena msgpack)
│                                  - Protocol encode/decode + key mapping
│                                  - Future: split protocol/embodiment/model concerns
│
├── serving_world.py             → OmniServingWorld
│                                  - Business layer
│                                  - Build OmniDiffusionRequest with session state
│                                  - Call DiffusionEngine.step()
│                                  - Extract action from result
│
├── session_manager.py           → WorldSessionState, WorldSessionStore
│                                  - Session lifecycle
│                                  - CPU-side state (frame buffers, pointers)
│
└── protocol/world.py            → Protocol definitions
                                   - Request/response models
```

Naming rationale:
- `world` — covers world models broadly (DreamZero, interactive video, game sim)
- `stream` — aligns with `serving_speech_stream.py`, indicates WebSocket multi-turn
- Protocol layer (`serving_world_stream.py`) separated from business layer (`serving_world.py`) so multiple protocols (roboarena msgpack, JSON native) can share the same backend

---

## Discussion Points

### Point 1: API Endpoints

**Current DreamZero:** Single WebSocket endpoint, roboarena msgpack protocol.

**Proposal:** Two WebSocket endpoints sharing the same handler:

| Endpoint | Protocol | Client | Purpose |
|----------|----------|--------|---------|
| `/v1/world/openpi` | msgpack (roboarena compat) | `test_client_AR.py`, robot eval infra | Backward compat |
| `/v1/world/stream` | JSON (vllm-omni native) | New integrations | Future standard |

Additionally, REST endpoints for session management:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/world/sessions` | POST | Create session (optional, auto-created on first infer) |
| `/v1/world/sessions/{id}` | DELETE | Destroy session, trigger video save |

- **Status:** DECIDED
- **Decision:** MVP: `/v1/world/openpi` only (msgpack WebSocket, roboarena compat). P1: add `/v1/world/stream` (JSON native). REST session endpoints deferred to P1.

### Point 2: Server Loop & Request Handling

- **Status:** DECIDED
- **Prerequisite PR:** "Multi-Turn Stateful Inference for Diffusion Engine" (`PR3_multi_turn_stateful_diffusion_engine.md`)
- **Decision:** Through DiffusionEngine. Session state metadata travels in `extra_args` on each request, returned via `custom_output`. Engine, pre_process_func, post_process_func all remain stateless. Handler owns the session store and does the state round-trip.
- **Note:** DiffusionEngine.step() is synchronous (unlike AR engine's async generator). Use `asyncio.to_thread` to avoid blocking the event loop.

**Comparison with TTS stream:**

| | TTS stream | DreamZero |
|---|---|---|
| Engine | AR engine: `engine_client.generate()` — async generator | DiffusionEngine: `engine.step(request)` — sync blocking |
| Loop pattern | `async for res in generator` | `await asyncio.to_thread(engine.step, request)` |
| Blocking? | No | No (via to_thread) |

**Server loop:**

```python
async def _handler(self, websocket):
    # 1. Send metadata (roboarena protocol)
    await websocket.send(packer.pack(server_config))

    # 2. Request-response loop
    while True:
        # 2a. Receive observation (async wait for client)
        data = await websocket.recv()
        obs = msgpack_numpy.unpackb(data)

        endpoint = obs.pop("endpoint", "infer")
        if endpoint == "reset":
            self._reset_session(obs)
            await websocket.send("reset successful")
            continue

        # 2b. Pack session state into request
        session = self._get_or_create_session(obs)
        request = self._build_request(obs, session)

        # 2c. Run through DiffusionEngine (sync → to_thread)
        result = await asyncio.to_thread(self.engine.step, request)

        # 2d. Update session state from response
        self._update_session(session, result)

        # 2e. Send action back to client
        action = self._convert_action(result)
        await websocket.send(packer.pack(action))
```

### Point 3: Session State — CPU Side

- **Status:** DECIDED
- **Decision:** `session_manager.py` provides base class `WorldSessionState` + `WorldSessionStore`. Model-specific state is a derived class. Handler uses store to manage sessions, packs state into `extra_args` per request.

**Base class** (`vllm_omni/entrypoints/openai/session_manager.py`):

```python
@dataclass
class WorldSessionState:
    """Base session state for all world models."""
    session_id: str
    is_first_call: bool = True
    call_count: int = 0
    current_start_frame: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def reset(self):
        """Reset state for new episode. Subclasses extend this."""
        self.is_first_call = True
        self.call_count = 0
        self.current_start_frame = 0


class WorldSessionStore:
    """Manages WorldSessionState instances with TTL cleanup."""
    def __init__(self, max_sessions=4, ttl_seconds=300.0): ...
    def get_or_create(self, session_id, state_cls=WorldSessionState) -> WorldSessionState: ...
    def destroy(self, session_id) -> bool: ...
    def reset(self, session_id): ...
    def cleanup_expired(self): ...
```

**DreamZero derived class** (`vllm_omni/diffusion/models/dreamzero/session_state.py`):

```python
@dataclass
class DreamZeroSessionState(WorldSessionState):
    """DreamZero-specific: 3 fixed cameras + video accumulator."""
    frame_buffers: dict[str, list[np.ndarray]] = field(default_factory=lambda: {
        "video.exterior_image_1_left": [],
        "video.exterior_image_2_left": [],
        "video.wrist_image_left": [],
    })
    video_accumulator: list[torch.Tensor] = field(default_factory=list)

    def reset(self):
        super().reset()
        for buf in self.frame_buffers.values():
            buf.clear()
        self.video_accumulator.clear()
```

**Usage in handler:**

```python
# DreamZero handler passes its state class to store
session = self.session_store.get_or_create(
    session_id, state_cls=DreamZeroSessionState
)
```

See `PR3_multi_turn_stateful_diffusion_engine.md` for the full state round-trip flow.

### Point 4: Session State — GPU Side (KV Cache)

- **Status:** DECIDED
- **Decision:** KV cache stays inside the model for MVP. Pipeline reads `current_start_frame` from `extra_args`, manages KV cache internally, returns updated `current_start_frame` via `custom_output`. Handler and pipeline must agree on frame pointer.
- **Future (P3):** Externalize KV cache to a diffusion block manager (analogous to vllm AR engine). See `PR3_multi_turn_stateful_diffusion_engine.md` § Future work.

**GPU state (model-internal):**

| State | Size | Managed by |
|-------|------|-----------|
| KV cache (cond + uncond) | ~2GB | Pipeline (auto-reset when context full) |
| Cross-attn cache | Small | Pipeline |
| `clip_feas`, `ys`, `language` | Tensors | Pipeline (cached after first encode) |

**Auto-reset triggers (inside pipeline):**
1. `language is None` → first call
2. Language changed → reset
3. Single frame input → reset
4. `current_start_frame >= local_attn_size` → context window full

**Client reset:** Handler sends `extra_args["reset"] = True` via normal `engine.step()` path. Pipeline checks flag at the start of `forward()`, clears all GPU state (KV cache, clip_feas, language, current_start_frame), returns early with empty output. No need for a separate `reset()` method — keeps handler decoupled from pipeline.

### Insight: vllm-omni Responsibility Split Pattern

All existing models follow the same split:

```
API/Protocol layer (api_server route / stream handler):
  - Protocol decode (JSON, msgpack, Form data)
  - Key mapping: protocol-specific keys → OmniTextPrompt / OmniDiffusionSamplingParams
  - Construct OmniDiffusionRequest
  - Result → protocol-specific encode (base64, wav bytes, msgpack)

Model layer (pre_process_func / pipeline.forward / post_process_func):
  - pre_process: image load, resize, VideoProcessor.preprocess
  - forward: text encode, image encode, VAE, denoising
  - post_process: tensor → numpy / video format
```

Evidence:
- `generate_images()` in api_server.py: parses `ImageGenerationRequest` (REST JSON) → builds `OmniTextPrompt` + `OmniDiffusionSamplingParams` → calls engine → base64 encodes result
- `serving_speech_stream.py`: parses WebSocket JSON → builds `OpenAICreateSpeechRequest` → calls speech service → sends audio bytes
- `serving_video.py`: parses `VideoGenerationRequest` (Form data) → builds params → calls engine → video encode

**Key principle:** Protocol-specific logic stays in API layer. Model-specific logic stays in pre/post process + pipeline. DreamZero follows the same pattern.

---

### Point 5: Observation Format Conversion

**Current DreamZero has two layers of conversion:**

```
roboarena format             →  AR_droid format              →  model input
(from client)                   (frame accumulation)            (normalized tensors)

observation/exterior_image_0     video.exterior_image_1          images: (B,C,T,H,W)
observation/joint_position       state.joint_position            state: (B,1,D)
prompt                           annotation.language.action_text text: tokenized
```

Layer 1 (`_convert_observation`): key mapping + frame buffer accumulation → in handler
Layer 2 (`apply()` in sim_policy): normalization, dtype conversion → in pipeline

- **Status:** DECIDED
- **Decision:** Follows the vllm-omni responsibility split pattern:

| What | Where | Why |
|------|-------|-----|
| Key mapping (`observation/exterior_image_0_left` → `cam_0`) | Handler (protocol layer) | Protocol-specific: roboarena uses 0-indexed keys, JSON native may use different keys |
| Frame accumulation + selection (buffer append, take last N) | pre_process_func (model layer) | Model-specific: DreamZero needs 1-frame for first call, 4-frame for subsequent |
| Resize, normalize, dtype conversion | pre_process_func (model layer) | Model-specific |
| Text encode, VAE encode, CLIP encode | pipeline.forward (model layer) | Model-specific |

**Flow:**
```
handler:           obs["observation/exterior_image_0_left"] → request.prompts[0]["multi_modal_data"]["cam_0"]
                   obs["prompt"] → request.prompts[0]["prompt"]
                   obs["session_id"] → request.extra_args["session_id"]
                   session.frame_buffers → request.extra_args["frame_buffers"]

pre_process_func:  extra_args["frame_buffers"]["cam_0"].append(new_frame)
                   selected = buffer[-num_frames:]
                   extra_args["selected_frames"] = selected

pipeline.forward:  read extra_args["selected_frames"], encode, denoise
```

Two protocols produce the same `OmniDiffusionRequest` format, share the same pre_process_func.

**Note on naming:** The handler currently mixes three concerns — protocol (roboarena msgpack), embodiment (DROID cameras/action space), and model (DreamZero). MVP uses a single `OmniWorldStreamHandler` that bakes in all three. Future work: separate protocol (roboarena vs JSON native) from embodiment config (DROID vs AgiBot) from model config, so the handler is composable. See Future Work section.

### Point 6: Action Format Conversion

**Current DreamZero:**
```
model output                    →  roboarena format
action.joint_position (N, 7)       np.ndarray (N, 8)
action.gripper_position (N, 1)     [joint + gripper concatenated]
```

**Proposal:**
- Handler converts model output → client format
- For roboarena: concatenate joint + gripper → `(N, 8)` array, msgpack encode
- For native protocol: return structured JSON with separate fields

- **Status:** DECIDED
- **Decision:** Handler converts `custom_output["actions"]` to protocol format. MVP: roboarena expects `(N, 8)` numpy array (7 joint + 1 gripper concatenated), msgpack encoded. P1 native protocol TBD.

### Point 7: Integration with Existing vllm-omni Components

**Which existing components can we reuse?**

| Component | Reuse? | Notes |
|-----------|--------|-------|
| `DiffusionEngine` | Yes | Confirmed in Point 2 — engine.step with extra_args for state |
| `CFGParallelMixin` | Yes | After prerequisite PRs (reduce broadcast, cfg_combine_mask) |
| `FlowUniPCMultistepScheduler` | Yes | Already in vllm-omni |
| `DistributedAutoencoderKLWan` | Yes | VAE for encode/decode |
| `ProgressBarMixin` | Yes | For denoising loop progress |
| `StepCache` | Yes | After StepCache PR |
| `api_server.py` router | Yes | Register WebSocket endpoint |
| `DiffusersPipelineLoader` | Maybe | Weight loading — depends on DreamZero checkpoint format |

- **Status:** DECIDED
- **Decision:** Follow existing pattern (same as Wan2.2 I2V):
  - `__init__`: text_encoder / image_encoder / VAE loaded via `from_pretrained` from Wan2.1 base model
  - `__init__`: `CausalWanModel` built from config (structure only, no weights)
  - `weights_sources`: points to DreamZero checkpoint, loads all `action_head.*` keys
  - `load_weights`: key remapping (`action_head.model.* → transformer.*`, `action_head.vae.* → vae.*`, etc.) to overlay DreamZero fine-tuned weights onto base components

### Point 8: Startup & Model Loading

**Current DreamZero startup:**
```
torchrun --nproc_per_node=8 socket_test_optimized_AR.py --port 8000
→ init_device_mesh (NCCL)
→ GrootSimPolicy(model_path, device_mesh)
  → load backbone (Wan2.1-I2V-14B)
  → load DreamZero action head weights
  → post_initialize (move to GPU, set dtype)
→ ARDroidRoboarenaPolicy(policy)
→ WebsocketPolicyServer.serve_forever()
```

**Proposal for vllm-omni:**
```
vllm serve <model> --omni --port 8091
→ existing vllm-omni startup
→ detect world model → create DreamZeroPipeline
→ register /v1/world/openpi WebSocket endpoint
→ serve
```

Or standalone:
```
python -m vllm_omni.entrypoints.openai.serving_world_stream \
    --model-path <path> --port 8000
```

- **Status:** DECIDED
- **Decision:** Integrated into `vllm serve --omni`. No standalone script needed. Startup chain:
  ```
  vllm serve <model> --omni
    → registry looks up "DreamZeroPipeline" (or "VLA")
    → DiffusersPipelineLoader creates pipeline + loads weights
    → api_server registers /v1/world/openpi WebSocket route
    → OmniWorldStreamHandler(engine) created
    → serve
  ```
  Multi-GPU: uses vllm-omni's existing distributed init (DiffusionExecutor), not `torchrun`.

---

## TODO

Discuss each point, then mark decision. Implementation order follows decisions.

| Point | Topic | Status |
|-------|-------|--------|
| 1 | API Endpoints | DECIDED — MVP: `/v1/world/openpi` only |
| 2 | Server Loop & Request Handling | DECIDED — through DiffusionEngine, state in extra_args |
| 3 | Session State — CPU Side | DECIDED — base `WorldSessionState` + `DreamZeroSessionState` derived, `WorldSessionStore` in `session_manager.py` |
| 4 | Session State — GPU Side (KV Cache) | DECIDED — model-internal, reset via `extra_args["reset"]` flag through engine.step |
| 5 | Observation Format Conversion | DECIDED — key mapping in handler, frame accumulation + normalize in pre_process_func |
| 6 | Action Format Conversion | DECIDED — handler converts, roboarena returns (N,8) numpy |
| 7 | Integration with Existing Components | DECIDED — same pattern as Wan2.2 I2V, weights_sources + key remapping in load_weights |
| 8 | Startup & Model Loading | DECIDED — `vllm serve --omni`, no standalone script, uses existing distributed init |

---

## Future Work

### Extend OmniWorldStreamHandler to support all scenarios

MVP's `OmniWorldStreamHandler` bakes in three concerns:
1. **Protocol** — roboarena msgpack WebSocket
2. **Embodiment** — DROID (3 cameras, 8-dim action, 180x320 resolution)
3. **Model** — DreamZero (frame schedule, action horizon)

Future: decompose into composable layers:

```
OmniWorldStreamHandler (base)
  - Session lifecycle (connect, loop, reset, disconnect)
  - Engine call (build request, call engine.step, extract result)
  │
  ├── Protocol layer (swappable)
  │   ├── RoboarenaProtocol — msgpack encode/decode
  │   └── NativeProtocol — JSON encode/decode
  │
  ├── Embodiment config (data-driven)
  │   ├── DROID: 3 cameras, joint_position action space, (180,320)
  │   ├── AgiBot: different cameras, different action dims
  │   └── loaded from config file or model metadata
  │
  └── Model config (from pipeline)
      ├── DreamZero: frame schedule, action horizon, num_frame_per_block
      └── Future world models: different observation/action schemas
```

This enables:
- Same handler serves DreamZero on DROID, DreamZero on AgiBot, or a different world model
- Adding a new protocol = one new encode/decode class, no handler changes
- Adding a new embodiment = one new config, no code changes
