# DreamZero Integration Summary

Master document for DreamZero (World Model) support in vllm-omni.

---

## 1. Document Index

| Document | Type | Description |
|----------|------|-------------|
| [`dreamzero_summary.md`](dreamzero_summary.md) | Summary | This file — dependency graph, class diagram, data flow |
| [`dreamzero_cfg_parallel_adaptation.md`](dreamzero_cfg_parallel_adaptation.md) | Design | CFG parallel adaptation: 6 diffs, all decided |
| [`dreamzero_serving_adaptation.md`](dreamzero_serving_adaptation.md) | Design | Serving architecture: async closed-loop, TTS comparison |
| [`dreamzero_server_implementation.md`](dreamzero_server_implementation.md) | Design | Server implementation: 8 points, all decided (OpenPI WebSocket) |
| [`unified_world_model_api_analysis.md`](unified_world_model_api_analysis.md) | Analysis | DreamZero vs OpenPI vs LeRobot API comparison |
| [`PR1_reduce_redundant_broadcast_cfg_parallel.md`](PR1_reduce_redundant_broadcast_cfg_parallel.md) | PR spec | Remove redundant broadcast, add cfg_combine_mask |
| [`PR2_step_cache.md`](PR2_step_cache.md) | PR spec | Step-level prediction caching |
| [`PR3_multi_turn_stateful_diffusion_engine.md`](PR3_multi_turn_stateful_diffusion_engine.md) | PR spec | Session state via extra_args round-trip |
| [`PR4_lerobot_grpc_api.md`](PR4_lerobot_grpc_api.md) | PR spec | LeRobot gRPC AsyncInference service |
| [`PR5_async_pipeline.md`](PR5_async_pipeline.md) | PR spec | Speculative execution using predicted video |

---

## 2. Dependency Graph

```
 Infrastructure PRs (model-agnostic)         Serving PRs
 ─────────────────────────────────           ──────────────────────────────

 PR1: Reduce Broadcast    PR2: StepCache     PR3: Multi-Turn Stateful Engine
  - All ranks local        - Hook predict_     - State in extra_args
  - cfg_combine_mask         noise_maybe_cfg   - WorldSessionState/Store
       │                        │                       │
       └──────────┬─────────────┘                       │
                  │                                     │
                  ▼                                     │
        PR6: DreamZero Pipeline ◄───────────────────────┘
         - CausalWanModel
         - Action encoders/decoder
         - VideoActionScheduler
         - Registry + weight loading
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
 PR7: OpenPI WebSocket    PR4: LeRobot gRPC
  - /v1/world/openpi       - AsyncInference service
  - Serial recv→infer→send - Decoupled obs/action
  - roboarena compat       - LeRobot RobotClient compat
  - OpenPI client compat          │
                                  │
                                  ▼
                          PR5: Async Pipeline
                           - Speculative execution
                           - Predict video → use as next obs
                           - ~100% GPU utilization
                           - World model exclusive feature
```

### Implementation Order

| Step | Scope | Files | Depends On |
|------|-------|-------|------------|
| **PR1** | Reduce broadcast + cfg_combine_mask | `cfg_parallel.py` | None |
| **PR2** | StepCache | `cfg_parallel.py` | None (parallel with PR1) |
| **PR3** | Multi-turn stateful engine | Design only — uses existing `extra_args`/`custom_output` | None |
| **PR4** | LeRobot gRPC API | `serving_world_grpc.py` (new), proto files | PR3 |
| **PR5** | Async pipeline (speculative) | `serving_world_grpc.py` (extend) | PR4, PR6 |
| **PR6** | DreamZero pipeline | `diffusion/models/dreamzero/` (new), `registry.py` | PR1, PR2, PR3 |
| **PR7** | OpenPI WebSocket serving | `serving_world_stream.py` (new), `serving_world.py` (new), `session_manager.py` (new), `api_server.py` | PR3, PR6 |

**Parallelism:** PR1, PR2, PR3 are all standalone. PR4 and PR7 are independent (different protocols). PR5 requires both PR4 and PR6.

### Serving Protocol Comparison

| | PR7: OpenPI WebSocket | PR4: LeRobot gRPC | PR5: Async Pipeline |
|---|---|---|---|
| Protocol | WebSocket + msgpack | gRPC (protobuf + pickle) | gRPC (extends PR4) |
| Loop | Serial `recv → infer → send` | Decoupled `SendObs` / `GetActions` | Server-driven inference loop |
| Obs handling | Process in order | Queue(maxsize=1), overwrite old | Queue + speculative from video_pred |
| GPU utilization | ~50% | ~50% (client-driven) | ~100% (server-driven) |
| Compatible clients | test_client_AR, OpenPI, run_sim_eval | LeRobot RobotClient | LeRobot RobotClient |
| Models supported | Any world model | Any world model | World models with video prediction only |

### Future Work

| Item | Description | Depends On |
|------|-------------|------------|
| Native JSON protocol | `/v1/world/stream` endpoint | PR7 |
| REST session API | `POST /v1/world/sessions`, `DELETE` | PR7 |
| Handler decomposition | Separate protocol / embodiment / model | PR7 |
| KV cache externalization | Diffusion block manager, multi-session | PR6 |
| TeaCache extractor | Transformer-level caching (stacks on StepCache) | PR6 |
| Multi-embodiment | AgiBot, other robots | PR4, PR7 |

---

## 3. Class Diagram

### CFGParallelMixin (modified by PR1, PR2)

```
CFGParallelMixin (metaclass=ABCMeta)
│  Modified methods:
│  ├── predict_noise_maybe_with_cfg(..., cfg_combine_mask=None)  ← PR1
│  │   - All ranks compute combine locally
│  │   - StepCache check at entry/exit                           ← PR2
│  ├── scheduler_step_maybe_with_cfg(...)  ← PR1: remove broadcast
│  ├── predict_noise(**kwargs) → Tensor | tuple[Tensor, ...]
│  ├── combine_cfg_noise(pos, neg, scale)
│  ├── scheduler_step(pred, t, latents)
│  └── diffuse(...)
│  New attributes:
│  └── _step_cache: StepCache | None = None  ← PR2
```

### StepCache (PR2)

```
StepCache
├── should_skip() → bool
├── get_cached() → Tensor | tuple
├── _record(pred)
├── _compute_similarity(a, b) → float
└── reset()
```

### Session Management (PR3)

```
WorldSessionState (base, in session_manager.py)
│  session_id, is_first_call, call_count, current_start_frame
│  reset()
└── DreamZeroSessionState (derived)
    │  frame_buffers: dict (3 cameras)
    │  video_accumulator: list[Tensor]
    │  reset()

WorldSessionStore
├── get_or_create(session_id, state_cls)
├── destroy(session_id)
├── reset(session_id)
└── cleanup_expired()
```

### DreamZero Pipeline (PR6)

```
DreamZeroPipeline(nn.Module, CFGParallelMixin, SupportImageInput, ProgressBarMixin)
│  Components: text_encoder, image_encoder, vae, transformer, scheduler
│  Overrides: predict_noise() → tuple, diffuse(), load_weights()
│
CausalWanModel(nn.Module)
│  40 CausalWanAttentionBlock, action encoder/decoder
│  GPU state: KV cache, clip_feas, ys, language, current_start_frame
│
VideoActionScheduler
│  Wraps video + action schedulers, step() → tuple
```

### Serving Layer

```
PR7: OpenPI WebSocket                     PR4: LeRobot gRPC
─────────────────────                     ────────────────────
api_server.py                             serving_world_grpc.py
  @router.websocket("/v1/world/openpi")     VLLMOmniPolicyServer(AsyncInferenceServicer)
                                              ├── Ready()
OmniWorldStreamHandler                       ├── SendPolicyInstructions()
  ├── handle_session(websocket)              ├── SendObservations() → obs_queue
  ├── _decode/_encode (msgpack)              ├── GetActions() → engine.step → actions
  └── serial: recv → engine.step → send      └── obs_queue: Queue(maxsize=1)

                Both share:
                ├── OmniServingWorld (business layer)
                ├── WorldSessionStore (session_manager.py)
                └── DiffusionEngine.step()


PR5: Async Pipeline (extends PR4)
─────────────────────────────────
VLLMOmniPolicyServer
  └── _inference_loop() (dedicated thread)
       ├── real obs available → infer(real)
       ├── no obs, has video_pred → infer(predicted)  ← speculative!
       └── result → action_queue + store video_pred
```

### Data Flow: OpenPI WebSocket (PR7)

```
Client (test_client_AR / OpenPI)
  │ msgpack({obs, session_id, endpoint="infer"})
  ▼
OmniWorldStreamHandler  →  recv → engine.step → send (serial)
  ▼
DiffusionEngine.step(request) → pre_process → pipeline.forward → post_process
  ▼
Client receives action (msgpack)
```

### Data Flow: LeRobot gRPC (PR4)

```
LeRobot RobotClient
  Thread 1 (control_loop):  [execute action] [send obs if buffer low]
  Thread 2 (recv_actions):  [GetActions → receive action chunk]
       │                          │
       ▼                          ▼
VLLMOmniPolicyServer
  SendObservations → obs_queue(maxsize=1, overwrite old)
  GetActions → take latest obs → engine.step → return actions
```

### Data Flow: Async Pipeline (PR5)

```
LeRobot RobotClient (same as PR4, no change)
       │
       ▼
VLLMOmniPolicyServer
  SendObservations → obs_queue
  GetActions → take from action_queue (pre-computed, near-instant)
  _inference_loop (server-driven, continuous):
       obs_queue has obs? → infer(real_obs) → action_queue + video_pred
       obs_queue empty?   → infer(video_pred as obs) → action_queue + new video_pred
       real obs arrives   → discard speculative, correct with real
```
