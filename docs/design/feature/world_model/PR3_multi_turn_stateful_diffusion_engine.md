# Multi-Turn Stateful Inference for Diffusion Engine

**PR Scope:** Enable DiffusionEngine to support stateful multi-turn inference, where session metadata travels with the request and the pipeline can maintain GPU-resident state (KV cache) across calls.

---

## Motivation

Existing DiffusionEngine assumes **stateless single-shot inference**: each `step(request)` is independent, `pre_process_func` and `post_process_func` are pure functions, and the pipeline's `forward()` has no memory of previous calls.

DreamZero (and future world models / AR diffusion models) requires **stateful multi-turn inference**:
- Client sends a sequence of observations over time
- Server accumulates state: frame buffers (CPU), KV cache (GPU), frame pointers
- Each inference call builds on the state from previous calls
- Session can be explicitly reset

This is analogous to vllm's AR engine where `Request` carries `kv_transfer_params`, `block_hashes`, and `cache_salt` to reference persistent KV cache state managed by the block manager.

---

## Design Principles

1. **State metadata travels with the request** — not hidden in closures or model internals
2. **pre_process_func and post_process_func remain pure functions** — read state from request, write updated state to response
3. **GPU-resident state (KV cache) stays in the model for now** — externalizing to a block manager is future work
4. **Backward compatible** — existing stateless pipelines see no change

---

## What Changes

### 1. `OmniDiffusionSamplingParams.extra_args` as session state carrier

Already exists. No structural change needed. DreamZero uses it to pass session metadata:

```python
request.sampling_params.extra_args = {
    # Session identity
    "session_id": "abc-123",

    # CPU-side state (carried per request, updated per call)
    "frame_buffers": {
        "cam_0": [np.ndarray, ...],  # accumulated frames
        "cam_1": [...],
        "wrist": [...],
    },
    "is_first_call": True,
    "call_count": 0,

    # GPU-side state pointer (model manages actual KV cache internally)
    "current_start_frame": 0,
}
```

### 2. `DiffusionOutput.custom_output` as state return channel

Already exists. Pipeline returns updated state metadata via `custom_output`:

```python
return DiffusionOutput(
    output=video_chunk,
    custom_output={
        "session_id": "abc-123",
        "actions": action_array,
        "current_start_frame": 5,  # updated pointer
        "chunk_index": 2,
    },
)
```

### 3. Session state round-trip flow

```
                        WebSocket Handler
                        (owns session store)
                              │
                    ┌─────────▼─────────────┐
                    │  Read session state     │
                    │  Pack into extra_args   │
                    └─────────┬─────────────┘
                              │
                    ┌─────────▼─────────────┐
                    │  OmniDiffusionRequest   │
                    │  .sampling_params       │
                    │    .extra_args = {      │
                    │      session_id,        │
                    │      frame_buffers,     │
                    │      is_first_call,     │
                    │      ...                │
                    │    }                    │
                    └─────────┬─────────────┘
                              │
              DiffusionEngine.step(request)
                              │
                    ┌─────────▼─────────────┐
                    │  pre_process_func       │
                    │  (stateless)            │
                    │  - Read frame_buffers   │
                    │    from extra_args      │
                    │  - Accumulate new frame │
                    │  - Write back to        │
                    │    extra_args           │
                    └─────────┬─────────────┘
                              │
                    ┌─────────▼─────────────┐
                    │  pipeline.forward()     │
                    │  - Read extra_args      │
                    │  - Use internal KV      │
                    │    cache (GPU)          │
                    │  - Return DiffusionOutput│
                    │    with custom_output   │
                    └─────────┬─────────────┘
                              │
                    ┌─────────▼─────────────┐
                    │  post_process_func      │
                    │  (stateless)            │
                    └─────────┬─────────────┘
                              │
                    ┌─────────▼─────────────┐
                    │  WebSocket Handler      │
                    │  - Read custom_output   │
                    │  - Update session store │
                    │  - Send action to client│
                    └─────────────────────────┘
```

### 4. WebSocket handler as session state owner

The handler (not pre_process_func, not model) owns the session store:

```python
class OmniWorldSessionHandler:
    def __init__(self):
        self._sessions: dict[str, WorldSessionState] = {}

    async def handle_infer(self, websocket, obs):
        session = self._get_or_create_session(obs["session_id"])

        # Pack session state into request
        request = OmniDiffusionRequest(
            prompts=[...],
            sampling_params=OmniDiffusionSamplingParams(
                extra_args={
                    "session_id": session.session_id,
                    "frame_buffers": session.frame_buffers,
                    "is_first_call": session.is_first_call,
                    "current_start_frame": session.current_start_frame,
                },
            ),
        )

        # Engine processes request — pre_process, pipeline, post_process all stateless
        result = self.engine.step(request)

        # Update session state from response
        custom = result.custom_output
        session.current_start_frame = custom["current_start_frame"]
        session.is_first_call = False
        session.call_count += 1

        return custom["actions"]
```

### 5. pre_process_func: stateless frame accumulation

```python
def get_dreamzero_pre_process_func(od_config):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        extra = request.sampling_params.extra_args
        frame_buffers = extra.get("frame_buffers", {})
        is_first_call = extra.get("is_first_call", True)

        # Read new frames from request
        multi_modal = request.prompts[0].get("multi_modal_data", {})
        for cam_key, frames in multi_modal.get("images", {}).items():
            if cam_key not in frame_buffers:
                frame_buffers[cam_key] = []
            frame_buffers[cam_key].append(frames)

        # Select frames based on call state
        num_frames = 1 if is_first_call else 4
        selected = {}
        for cam_key, buf in frame_buffers.items():
            selected[cam_key] = buf[-num_frames:]

        # Write back (handler will persist on next round-trip)
        extra["frame_buffers"] = frame_buffers
        extra["selected_frames"] = selected
        return request

    return pre_process_func
```

No closure state. All state read from and written to `extra_args`.

---

## GPU State Management

### Current scope (this PR)

KV cache stays inside the model as instance variables:
- `pipeline.transformer.kv_cache_cond` / `kv_cache_uncond`
- `pipeline.transformer.current_start_frame`
- `pipeline.transformer.clip_feas`, `ys`, `language`

Pipeline reads `current_start_frame` from `extra_args` to sync with the handler's session state. Internally, the model's `current_start_frame` and the handler's must agree — pipeline updates its internal pointer and returns the new value via `custom_output`.

### Constraint

**Single session per model instance.** Since KV cache is model-global, only one session can use a model at a time. The handler enforces this — rejects concurrent sessions or queues them.

### Future work (P3)

Externalize KV cache to a diffusion block manager:
- Session handler holds KV cache references (like AR engine's block table)
- Pipeline receives KV cache blocks via request, returns updated blocks
- Enables multi-session on same model instance via cache swapping
- Analogous to vllm AR engine's `kv_transfer_params` + block manager

---

## Backward Compatibility

| Component | Change | Impact on existing models |
|-----------|--------|--------------------------|
| `OmniDiffusionSamplingParams.extra_args` | No change (already exists) | None |
| `DiffusionOutput.custom_output` | No change (already exists) | None |
| `DiffusionEngine.step()` | No change | None |
| `pre_process_func` signature | No change (`request → request`) | None — existing funcs don't read extra_args |
| `post_process_func` signature | No change | None |
| `pipeline.forward()` | Model-specific — DreamZero reads extra_args | Other models don't read extra_args |

**Zero changes to DiffusionEngine or any existing model.** The stateful pattern is opt-in: only models that read `extra_args` participate.

---

## Relationship to Other PRs

| PR | Relationship |
|----|-------------|
| Reduce redundant broadcast | Independent — CFG parallel optimization |
| StepCache | Independent — prediction caching |
| cfg_combine_mask | Independent — tuple output support |
| DreamZero pipeline | Depends on this PR — uses extra_args for session state |
| DreamZero serving | Depends on this PR — handler manages session store |
