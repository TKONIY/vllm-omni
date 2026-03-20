# DreamZero Serving Adaptation: Stateful Real-Time World Model API

Design document for adapting DreamZero's serving layer into vllm-omni, with focus on multi-turn stateful real-time inference.

---

## Reference Implementations

| | DreamZero (current) | vllm-omni TTS stream (reference) |
|---|---|---|
| **File** | `~/code/dreamzero/socket_test_optimized_AR.py` | `vllm_omni/entrypoints/openai/serving_speech_stream.py` |
| **Protocol** | roboarena WebSocket (msgpack binary) | OpenAI-style WebSocket (JSON text + binary audio) |
| **Session** | `session_id` in observation dict | `session.config` message at connection start |
| **State** | Frame buffers, KV cache, `current_start_frame`, `is_first_call` | Stateless per sentence (SentenceSplitter buffer only) |
| **Multi-turn** | Yes — same session sends 15+ sequential observations, state accumulates | No — each sentence is independent |
| **Reset** | Explicit `reset` endpoint from client | `input.done` ends session |
| **Latency target** | <143ms per chunk (7Hz real-time) | Not latency-critical |

---

## Asynchronous Closed-Loop Execution

DreamZero achieves real-time robot control (7Hz) not through async code, but through **action chunking + open-loop execution**. Understanding this is critical for serving design.

### How it works

**Source:** `~/code/dreamzero/eval_utils/run_sim_eval.py` — `DreamZeroJointPosClient`

```
Server returns action_horizon=24 actions per inference call.
Client only executes open_loop_horizon=8 actions before re-querying.

Timeline:
────────────────────────────────────────────────────────────
step 0:  client.infer(obs₀) → server (takes ~143ms)
         ← actions[0..23] (24 actions)
         robot executes actions[0]

step 1:  actions[1] from buffer (no server call, ~0ms)
step 2:  actions[2] from buffer
...
step 7:  actions[7] from buffer  ← open_loop_horizon exhausted

step 8:  client.infer(obs₈) → server (latest observation)
         ← new actions[0..23]
         robot executes actions[0]
...
```

### Key implementation

```python
class DreamZeroJointPosClient:
    def __init__(self, open_loop_horizon=8):
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None  # (N, 8) buffer

    def infer(self, obs, instruction):
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            # Buffer empty → query server
            result = self.client.infer(request_data)
            self.pred_action_chunk = result  # (24, 8)
            self.actions_from_chunk_completed = 0

        # Pop one action from buffer
        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1
        return action
```

### Implications for vllm-omni

1. **Server is stateless w.r.t. action buffering.** The server returns all 24 actions per call. The client decides how many to execute before re-querying. Server doesn't need to know about `open_loop_horizon`.

2. **Server inference latency is amortized.** 143ms for one server call covers 8 robot control steps at 7Hz (each step ~143ms in robot time). The math: `143ms server latency / 8 open-loop steps = 18ms amortized per step`. As long as server latency < `open_loop_horizon × control_period`, the robot never stalls.

3. **`test_client_AR.py` is synchronous** — it's a test script, not a real robot client. It calls `infer()` blocking every time. The real closed-loop client (`DreamZeroJointPosClient` in `run_sim_eval.py`) does the buffering.

4. **No async/threading needed on the server.** The "asynchronous" in "Asynchronous Closed-Loop Execution" refers to the **decoupling of prediction and execution** via action chunking — not async I/O. The server processes one request at a time, synchronously.

---

## Server-Side Inference: Reactive, Not Continuous

A common misconception: the server does **not** run a continuous inference loop. It is **reactive** — client sends observation, server runs once, returns actions. But KV cache **accumulates** across calls, giving the AR effect.

### Call frequency

- Robot control loop runs at 7Hz
- Client executes `open_loop_horizon=8` actions from buffer before re-querying
- Server call frequency: `7Hz / 8 = ~0.875Hz` (about once per second)
- Each call: ~143ms inference, returns 24 actions

### KV cache growth across calls

```
infer 0: KV cache = [frame_0]                              start_frame=1
infer 1: KV cache = [frame_0 | obs_1_frames]               start_frame=3
infer 2: KV cache = [frame_0 | obs_1 | obs_2]              start_frame=5
...
infer N: KV cache = [frame_0 | obs_1 | ... | obs_N]        start_frame=2N+1
         ↑ grows with each call (prefill appends, denoising only reads)

infer M: start_frame >= local_attn_size → AUTO-RESET
         KV cache recreated, start_frame=0, next call does fresh prefill
```

Each `infer` call does:
1. **Prefill** — VAE-encode new observation, run transformer with `update_kv_cache=True` to append to cache
2. **Denoise** — 4-step denoising loop with `update_kv_cache=False` (read-only)
3. **Advance pointer** — `current_start_frame += num_frame_per_block`

The KV cache is never partially replaced — it grows until the context window is full, then resets entirely.

### Implication for vllm-omni

The server can be a simple **request-response WebSocket handler** (like TTS stream). No background inference loop, no action streaming, no server-push. The "real-time" aspect is entirely handled by client-side action buffering.

---

## DreamZero Server-Side Inference Flow

```
Client connects via WebSocket
    │
    ├─ Server sends metadata (PolicyServerConfig as msgpack)
    │
    └─ Loop:
        Client sends observation (msgpack dict):
        │  - observation/exterior_image_{0,1}_left: (H,W,3) or (4,H,W,3)
        │  - observation/wrist_image_left: (H,W,3) or (4,H,W,3)
        │  - observation/joint_position: (7,)
        │  - observation/gripper_position: (1,)
        │  - prompt: str
        │  - session_id: str
        │  - endpoint: "infer" | "reset"
        │
        ├─ If "reset": clear all state, respond "reset successful"
        │
        └─ If "infer":
            │
            ├─ Session check: new session_id → reset state
            │
            ├─ _convert_observation:
            │   ├─ Accumulate frames into per-camera buffers
            │   ├─ First call: take 1 frame
            │   ├─ Subsequent: take last 4 frames
            │   └─ Convert state/prompt keys
            │
            ├─ Policy inference:
            │   ├─ Broadcast obs to worker ranks
            │   ├─ dist.barrier()
            │   ├─ lazy_joint_forward_causal(batch)
            │   │   ├─ apply() — normalize
            │   │   ├─ action_head.lazy_joint_video_action()
            │   │   │   ├─ Text encode (T5)
            │   │   │   ├─ Image encode (CLIP + VAE)
            │   │   │   ├─ KV cache init (if current_start_frame == 0)
            │   │   │   ├─ Prefill new frames into KV cache
            │   │   │   ├─ 4-step denoising loop (with CFG parallel)
            │   │   │   └─ Return (action_pred, video_pred)
            │   │   └─ unapply() — denormalize actions
            │   └─ dist.barrier()
            │
            ├─ _convert_action: dict → (N, 8) numpy array
            │
            └─ Send action back (msgpack)
```

### Stateful Components (persist across calls within a session)

| Component | Location | Lifecycle |
|-----------|----------|-----------|
| Frame buffers | `ARDroidRoboarenaPolicy._frame_buffers` | Accumulates across calls, cleared on reset |
| KV cache (cond) | `WANPolicyHead.kv_cache1` | Created at `current_start_frame==0`, grows with prefill, read during denoising |
| KV cache (uncond) | `WANPolicyHead.kv_cache_neg` | Same as above, for negative/unconditional branch |
| Cross-attn cache | `WANPolicyHead.crossattn_cache{,_neg}` | CLIP embeddings cached after first encode |
| Frame pointer | `WANPolicyHead.current_start_frame` | Increments each call, resets at 0 or when context window full |
| Language cache | `WANPolicyHead.language` | Cached text embeddings, reset on prompt change |
| CLIP features | `WANPolicyHead.clip_feas` | Cached after first image encode |
| VAE-encoded frames | `WANPolicyHead.ys` | Cached VAE latents of observation frames |
| First call flag | `ARDroidRoboarenaPolicy._is_first_call` | Controls 1-frame vs 4-frame input |
| Video accumulator | `ARDroidRoboarenaPolicy.video_across_time` | Stores video latents for periodic saving |

### Auto-Reset Triggers

The model resets `current_start_frame = 0` (and reinitializes KV cache) when:
1. `self.language is None` — first call ever
2. Language (prompt) changed between calls
3. `videos.shape[2] == 1` — single frame input (signals new session start)
4. `current_start_frame >= local_attn_size` — context window full

---

## vllm-omni TTS Stream Architecture

```
Client connects via WebSocket
    │
    ├─ Client sends {"type": "session.config", model, voice, ...}
    │   └─ Server validates config
    │
    └─ Loop:
        Client sends {"type": "input.text", "text": "..."}
        │
        ├─ SentenceSplitter accumulates text, emits complete sentences
        │
        └─ For each sentence:
            ├─ Server sends {"type": "audio.start", sentence_index, ...}
            ├─ Generate audio via OmniOpenAIServingSpeech
            ├─ Send audio bytes (binary frames)
            └─ Server sends {"type": "audio.done", sentence_index}
    │
    Client sends {"type": "input.done"}
    └─ Server sends {"type": "session.done"}
```

### Key differences from DreamZero

| Aspect | TTS Stream | DreamZero |
|--------|-----------|-----------|
| State complexity | Low — only text buffer | High — 10+ stateful components (KV caches, frame buffers, pointers) |
| Call pattern | Text in → audio out (one-shot per sentence) | Observation in → action out (iterative, state builds up) |
| Session semantics | Connection = session, `input.done` ends it | `session_id` field, explicit `reset`, connection can span sessions |
| GPU state | Stateless — each sentence is independent inference | Stateful — KV cache persists on GPU across calls |
| Concurrency | Can handle multiple sentences in flight | Single session per model instance (KV cache is model-global) |

---

## Design Challenges for vllm-omni Adaptation

### Challenge 1: GPU-Resident Session State

DreamZero's KV cache lives inside the model (`WANPolicyHead.kv_cache1`). This means:
- **Single session per GPU** — KV cache is model-global, not request-scoped
- **Cannot multiplex sessions** — unlike TTS where each request is stateless
- **Session affinity** — a session must always route to the same model instance

**Options:**
- **A. Single-session model:** Accept that one model instance serves one session at a time. Session management is at the routing layer (load balancer assigns robot to GPU).
- **B. Multi-session with explicit KV cache management:** Move KV cache out of the model into a session store. On each call, swap in the right session's KV cache. Adds latency from KV cache transfer.
- **C. Multi-model instances:** Run N model instances for N concurrent sessions. Each on separate GPU(s).

**Recommendation:** Option A for MVP — matches DreamZero's current design. Option C for scaling.

### Challenge 2: Frame Buffer Accumulation (CPU-side)

DreamZero accumulates frames in CPU-side buffers (`_frame_buffers`). The client sends 1 or 4 frames per call, and the server selects the last N frames.

In vllm-omni, observation preprocessing should happen **before the request enters the engine**. Options:
- **A. Serving layer manages frame buffers:** `OmniStreamingWorldHandler` accumulates frames, converts format, then creates `OmniDiffusionRequest` with the assembled observation.
- **B. Pipeline manages frame buffers:** Frame buffers live in the DreamZeroPipeline, passed via `extra_args`.

**Recommendation:** Option A — keeps the pipeline stateless (only KV cache is stateful), matches TTS stream pattern where the handler does preprocessing.

### Challenge 3: Protocol Design

DreamZero uses roboarena protocol (msgpack, custom keys). vllm-omni should offer a unified protocol that:
1. Works with `test_client_AR.py` (roboarena compat) — for existing robot eval infrastructure
2. Follows vllm-omni conventions (JSON-based, OpenAI-style) — for new integrations

**Recommendation:** Support both via two WebSocket endpoints:
- `/v1/world/stream` — vllm-omni native (JSON protocol, like TTS stream)
- `/v1/world/roboarena` — roboarena compat (msgpack protocol, for `test_client_AR.py`)

Both delegate to the same underlying handler.

### Challenge 4: Multi-Turn Session Lifecycle

TTS stream has a simple lifecycle: `config → text* → done`. DreamZero has a richer lifecycle:

```
Session lifecycle:
    connect → [metadata exchange]
    → infer (frame 0, 1 frame)         ← prefill, creates KV cache
    → infer (frames [0,7,15,23])        ← AR step 1, appends to KV cache
    → infer (frames [24,31,39,47])      ← AR step 2
    → ... (up to 15+ calls)
    → reset                             ← clears all state, saves video
    → (optionally start new session)
    → disconnect
```

State transitions:
```
IDLE → PREFILLED (after first infer)
     → RUNNING (after subsequent infers)
     → IDLE (after reset or context window full → auto-reset)
```

**Recommendation:** Model as `WorldModelSession` with explicit state machine:

```python
class WorldModelSession:
    state: Literal["idle", "prefilled", "running"]
    session_id: str
    frame_buffers: dict[str, list[np.ndarray]]
    is_first_call: bool
    video_accumulator: list[torch.Tensor]
    # KV cache stays in the model — session just tracks the pointer
    current_start_frame: int
```

### Challenge 5: Latency Requirements

DreamZero targets 7Hz (143ms per chunk). The vllm-omni diffusion engine's `add_req_and_wait_for_response` pattern adds overhead:
- Request queuing
- Pre-processing function dispatch
- Post-processing function dispatch

For real-time robotics, we may need a **fast path** that bypasses the queue and calls the pipeline directly — similar to how `socket_test_optimized_AR.py` calls `policy.lazy_joint_forward_causal(batch)` synchronously.

**Options:**
- **A. Direct pipeline call:** WebSocket handler calls pipeline.forward() directly, bypassing DiffusionEngine. Simplest, lowest latency. But loses engine features (scheduling, batching).
- **B. Priority queue in engine:** Add a real-time priority path in DiffusionEngine that skips queuing.
- **C. Dedicated executor:** Separate execution path for real-time world models.

**Recommendation:** Option A for MVP — DreamZero is single-session anyway, no batching benefit. The WebSocket handler holds a direct reference to the pipeline.

### Challenge 6: Distributed Inference Coordination

DreamZero uses `dist.barrier()` + manual broadcast for multi-GPU. vllm-omni has its own distributed execution model (DiffusionExecutor with workers).

**Options:**
- **A. Use DreamZero's distributed pattern:** WebSocket handler on rank 0, worker loop on other ranks, manual broadcast.
- **B. Use vllm-omni's DiffusionExecutor:** Route through engine, let executor handle distribution.
- **C. Hybrid:** WebSocket handler creates request, DiffusionExecutor handles multi-GPU.

**Recommendation:** Option A for MVP (matches current DreamZero). Option C for production integration.

---

## Proposed Protocol: `/v1/world/stream`

Follows vllm-omni conventions (JSON over WebSocket), modeled after TTS stream:

```
Client → Server:
    {"type": "session.config", "model": "...", "embodiment": "oxe_droid",
     "image_resolution": [180, 320], ...}
    {"type": "observation", "images": {...}, "state": {...}, "prompt": "..."}
    {"type": "observation", ...}  # repeated
    {"type": "session.reset"}     # explicit reset
    {"type": "session.destroy"}   # end session

Server → Client:
    {"type": "session.ready", "session_id": "...", "config": {...}}
    {"type": "action", "data": [[0.1, 0.2, ...]], "chunk_index": 0,
     "latency_ms": 120}
    {"type": "action", ...}
    {"type": "session.reset.done"}
    {"type": "session.destroyed", "video_saved": true}
    {"type": "error", "message": "..."}
```

### Proposed Protocol: `/v1/world/roboarena`

Exact roboarena protocol (msgpack over WebSocket) for `test_client_AR.py` compatibility:

```
Server → Client: msgpack(PolicyServerConfig dict)     # on connect
Client → Server: msgpack(obs dict with endpoint field) # infer or reset
Server → Client: msgpack(action array or "reset successful")
```

---

## Proposed Architecture

```
                            ┌──────────────────────────┐
                            │     WebSocket Layer       │
                            │                          │
test_client_AR ──msgpack──→ │  /v1/world/roboarena     │
                            │    RoboarenaProtocol     │
                            │         │                │
new clients ────JSON─────→  │  /v1/world/stream        │
                            │    NativeProtocol        │
                            │         │                │
                            └─────────┼────────────────┘
                                      │
                            ┌─────────▼────────────────┐
                            │  OmniWorldSessionHandler  │
                            │                          │
                            │  - Session store          │
                            │  - Frame buffer mgmt      │
                            │  - Observation conversion  │
                            │  - Action conversion       │
                            └─────────┼────────────────┘
                                      │
                            ┌─────────▼────────────────┐
                            │   DreamZeroPipeline       │
                            │   (CFGParallelMixin)      │
                            │                          │
                            │  - CausalWanModel         │
                            │  - KV cache (GPU)         │
                            │  - Text/Image encoders    │
                            │  - VAE                    │
                            │  - VideoActionScheduler   │
                            └──────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Stateful? |
|-----------|---------------|-----------|
| `RoboarenaProtocol` | msgpack encode/decode, roboarena key mapping | No |
| `NativeProtocol` | JSON encode/decode, vllm-omni key mapping | No |
| `OmniWorldSessionHandler` | Session lifecycle, frame buffers, obs/action conversion | Yes (CPU) |
| `DreamZeroPipeline` | Model inference, KV cache, denoising loop | Yes (GPU) |

---

## Implementation Plan

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| MVP | Single-session direct pipeline call, roboarena protocol only, single GPU | CFG parallel PRs (reduce broadcast, cfg_combine_mask) |
| P1 | Add native JSON protocol, StepCache acceleration | StepCache PR |
| P2 | Multi-GPU via DiffusionExecutor, priority scheduling | vllm-omni executor changes |
| P3 | Multi-session support (KV cache swapping or multi-instance) | Session management infra |

---

## Open Questions

1. **Should DreamZeroPipeline own the KV cache, or should it be externalized to the session handler?** Current DreamZero has it inside the model. Externalizing enables multi-session but adds complexity.

2. **How to handle `test_client_AR.py`'s frame schedule?** The client sends specific frame indices (`[0,7,15,23]`, `[24,31,39,47]`, ...). The server accumulates and selects. Should vllm-omni's native protocol require the client to manage frame selection, or should the server handle it?

3. **Video saving on reset** — DreamZero decodes accumulated video latents and saves as MP4 on reset. Should vllm-omni support this? If so, where does the decoding happen (pipeline or handler)?

4. **Connection vs session semantics** — TTS stream ties session to WebSocket connection. DreamZero uses `session_id` which can span connections. Which model for vllm-omni?
