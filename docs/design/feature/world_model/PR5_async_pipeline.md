# PR5: Async Pipeline for World Model Serving

**PR Scope:** Implement speculative execution in the gRPC serving layer — server continuously infers using predicted video frames when no real observation is available, achieving ~100% GPU utilization and near-zero action latency.

---

## Motivation

In the current serial serving model (OpenPI WebSocket and LeRobot gRPC), the server **waits for client to send observations** before inferring. With DreamZero's ~143ms inference time and client executing 8 action steps at ~18ms each:

```
Serial:   [infer 143ms][idle ~144ms][infer 143ms][idle ~144ms]...
           GPU util: ~50%
```

DreamZero is a **world model** — it predicts both actions AND future video frames. The predicted video can serve as a "virtual observation" for the next inference step, without waiting for the real observation from the robot.

```
Async:    [infer(real_obs)][infer(pred_video)][infer(pred_video)][infer(real_obs)]...
           GPU util: ~100%, actions always pre-computed in queue
```

This is only possible for world models (DreamZero), not standard VLAs (Pi0, ACT) which don't produce video predictions.

## Design

### Server Inference Loop

Add a dedicated inference thread to `VLLMOmniPolicyServer` (from PR4):

```python
class VLLMOmniPolicyServer(services_pb2_grpc.AsyncInferenceServicer):

    def __init__(self, engine, session_store):
        self.obs_queue = Queue(maxsize=1)      # real obs from client
        self.action_queue = Queue(maxsize=1)    # pre-computed actions for client
        self.last_video_pred = None             # predicted video for speculative exec
        self.inference_thread = Thread(target=self._inference_loop, daemon=True)

    def _inference_loop(self):
        """Continuously infer — use real obs when available, predicted video otherwise."""
        while self.running:
            if not self.obs_queue.empty():
                # Real observation available → use it, discard speculative state
                obs = self.obs_queue.get()
                self.last_video_pred = None  # reset speculation
                source = "real"
            elif self.last_video_pred is not None:
                # No real obs → use predicted video as virtual obs (speculative)
                obs = self._video_pred_to_obs(self.last_video_pred)
                source = "predicted"
            else:
                # Nothing available → wait for first real obs
                obs = self.obs_queue.get()
                source = "real"

            result = self.engine.step(self._build_request(obs, source))

            # Store video prediction for next speculative step
            self.last_video_pred = result.custom_output.get("video_pred")

            # Put actions in queue (overwrite old)
            if self.action_queue.full():
                self.action_queue.get_nowait()
            self.action_queue.put(result)

    def GetActions(self, request, context):
        """Client gets pre-computed actions — near-instant return."""
        result = self.action_queue.get(timeout=timeout)
        return Actions(data=pickle.dumps(result))
```

### Correction on Real Observation

When a real observation arrives after speculative steps:

```
t=0:   real obs_0 → infer → action_0 + video_pred_0
t=143: video_pred_0 → infer → action_1 + video_pred_1   (speculative)
t=286: video_pred_1 → infer → action_2 + video_pred_2   (speculative)
t=300: real obs_1 arrives!
       → discard video_pred_2
       → real obs_1 → infer → action_3 + video_pred_3   (corrected)
```

The KV cache from speculative steps may need to be rolled back when real obs arrives. Two strategies:

**A. Full reset:** When real obs arrives, reset KV cache to last real-obs checkpoint. Simple but wastes speculative KV.

**B. Checkpoint + rollback:** Save KV cache state at each real obs. Speculative steps build on it. When new real obs arrives, restore checkpoint and re-prefill. More efficient but complex.

MVP uses strategy A. Strategy B is future optimization.

### Pipeline.forward Changes

Pipeline needs to support speculative input via `extra_args`:

```python
extra_args = {
    "source": "real" | "predicted",    # NEW: tells pipeline the obs source
    "video_pred": tensor,               # NEW: predicted video from last step
    ...
}
```

When `source == "predicted"`:
- Skip CLIP/VAE encoding (video_pred is already in latent space)
- Prefill KV cache with predicted latent directly
- Mark speculative in session state (for potential rollback)

### DiffusionOutput Extension

Pipeline returns video prediction for the inference loop to use:

```python
return DiffusionOutput(
    output=video_chunk,
    custom_output={
        "actions": action_array,
        "video_pred": video_latent,     # NEW: for speculative next step
        "source": "real" | "predicted",
        "current_start_frame": ...,
    },
)
```

## Performance Impact

| Metric | Serial (PR5 OpenPI) | Async Pipeline (this PR) |
|--------|-------------------|--------------------------|
| GPU utilization | ~50% | ~100% |
| Action latency | 143ms (wait for infer) | ~0ms (pre-computed in queue) |
| Obs-to-action delay | 143ms | 143ms (first real), ~0ms (speculative) |
| Accuracy | Exact | Slightly reduced during speculative steps (corrected on real obs) |

## Dependencies

- PR4 (LeRobot gRPC API) — provides the decoupled obs/action gRPC service
- PR6 (DreamZero pipeline) — pipeline must return `video_pred` in `custom_output`

## Limitations

- Only works for **world models** that predict future video (DreamZero). Standard VLAs (Pi0, ACT) cannot do speculative execution.
- Speculative actions may diverge from reality — corrected when real obs arrives, but robot may have executed divergent actions in the meantime.
- KV cache rollback strategy needs tuning per model.
