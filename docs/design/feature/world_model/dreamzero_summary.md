# DreamZero Integration Summary

Master document for DreamZero (World Model) support in vllm-omni.

---

## 1. Document Index

| Document | Type | Description |
|----------|------|-------------|
| [`dreamzero_summary.md`](dreamzero_summary.md) | Summary | This file — overview, dependency graph, class diagram |
| [`dreamzero_cfg_parallel_adaptation.md`](dreamzero_cfg_parallel_adaptation.md) | Design | CFG parallel adaptation: 6 diffs, all decided |
| [`dreamzero_serving_adaptation.md`](dreamzero_serving_adaptation.md) | Design | Serving architecture: data flow, async closed-loop, TTS comparison |
| [`dreamzero_server_implementation.md`](dreamzero_server_implementation.md) | Design | Server implementation: 8 points, all decided |
| [`PR1_reduce_redundant_broadcast_cfg_parallel.md`](PR1_reduce_redundant_broadcast_cfg_parallel.md) | PR spec | Remove redundant broadcast, add cfg_combine_mask tuple support |
| [`PR2_step_cache.md`](PR2_step_cache.md) | PR spec | Step-level prediction caching in CFGParallelMixin |
| [`PR3_multi_turn_stateful_diffusion_engine.md`](PR3_multi_turn_stateful_diffusion_engine.md) | PR spec | Session state via extra_args round-trip |

---

## 2. Dependency Graph

```
PR1: Reduce Redundant Broadcast          PR2: StepCache          PR3: Multi-Turn Stateful Engine
 - All ranks compute locally              - Hook in predict_      - State in extra_args
 - Remove broadcast                         noise_maybe_with_cfg  - custom_output return
 - Add cfg_combine_mask                   - Cosine similarity     - WorldSessionState base
   ("cfg"/"positive"/"negative")            skip                  - WorldSessionStore
         │                                      │                         │
         │                                      │                         │
         └──────────────┬───────────────────────┘                         │
                        │                                                 │
                        ▼                                                 │
              DreamZero Pipeline                                          │
               - CausalWanModel (transformer)                             │
               - Action encoders/decoder                                  │
               - VideoActionScheduler                                     │
               - DreamZeroSessionState                                    │
               - TeaCache extractor (future)                              │
               - Registry + weight loading                                │
                        │                                                 │
                        └─────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                                    DreamZero Serving
                                     - OmniWorldStreamHandler
                                     - OmniServingWorld
                                     - session_manager.py
                                     - /v1/world/roboarena route
```

### Implementation Order

| Step | Scope | Files Changed | Depends On |
|------|-------|--------------|------------|
| **PR1** | Reduce broadcast + cfg_combine_mask | `cfg_parallel.py` | None (standalone) |
| **PR2** | StepCache | `cfg_parallel.py` | None (standalone, can parallel with PR1) |
| **PR3** | Multi-turn stateful engine | No file changes — uses existing `extra_args`/`custom_output` | None (standalone) |
| **PR4** | DreamZero pipeline | `diffusion/models/dreamzero/` (new), `registry.py` | PR1, PR2, PR3 |
| **PR5** | DreamZero serving | `entrypoints/openai/serving_world*.py` (new), `session_manager.py` (new), `api_server.py` | PR4 |

### Future Work

| Item | Description | Depends On |
|------|-------------|------------|
| Native JSON protocol | `/v1/world/stream` endpoint | PR5 |
| REST session API | `POST /v1/world/sessions`, `DELETE /v1/world/sessions/{id}` | PR5 |
| Handler decomposition | Separate protocol / embodiment / model concerns | PR5 |
| KV cache externalization | Diffusion block manager, multi-session on same GPU | PR4 |
| TeaCache extractor | Additional **transformer-level** caching that stacks on top of StepCache. StepCache (PR2) already covers DreamZero's `should_run_model` (step-level skip, no extractor needed). TeaCache extractor adds a second layer: when StepCache decides not to skip, TeaCache can still skip transformer blocks via residual reuse. Requires writing an extractor to split `CausalWanModel.forward` into preprocess / modulated_input / blocks / postprocess. | PR4 |
| Multi-embodiment | AgiBot, other robots | PR5 |

---

## 3. Class Diagram

### CFGParallelMixin (modified by PR1, PR2)

```
CFGParallelMixin (metaclass=ABCMeta)
│
│  Modified methods:
│  ├── predict_noise_maybe_with_cfg(
│  │       ...,
│  │       cfg_combine_mask=None,          ← PR1: tuple output support
│  │   )
│  │   - All ranks compute combine locally  ← PR1: remove rank-0-only logic
│  │   - StepCache check at entry/exit      ← PR2: skip if similar
│  │
│  ├── scheduler_step_maybe_with_cfg(...)
│  │   - Remove broadcast                   ← PR1: all ranks step locally
│  │
│  ├── predict_noise(**kwargs) → Tensor | tuple[Tensor, ...]
│  ├── combine_cfg_noise(pos, neg, scale)
│  ├── cfg_normalize_function(pred, combined)
│  ├── scheduler_step(pred, t, latents)
│  └── diffuse(...)                         # subclasses implement
│
│  New attributes:
│  └── _step_cache: StepCache | None = None  ← PR2
```

### StepCache (PR2)

```
StepCache
├── similarity_fn: Callable
├── threshold: float
├── warmup_steps: int
├── max_skip_steps: int
├── _cache: Tensor | tuple | None
├── _step_count: int
├── _consecutive_skips: int
│
├── should_skip() → bool
├── get_cached() → Tensor | tuple
├── _record(pred)
├── _compute_similarity(a, b) → float
└── reset()
```

### Session Management (PR3 + PR5)

```
WorldSessionState (base, in session_manager.py)
│  session_id: str
│  is_first_call: bool
│  call_count: int
│  current_start_frame: int
│  created_at: float
│  last_accessed: float
│  reset()
│
└── DreamZeroSessionState (derived, in diffusion/models/dreamzero/)
    │  frame_buffers: dict[str, list[np.ndarray]]   # 3 fixed cameras
    │  video_accumulator: list[torch.Tensor]
    │  reset()  # extends base
    │
WorldSessionStore (in session_manager.py)
    ├── get_or_create(session_id, state_cls) → WorldSessionState
    ├── destroy(session_id) → bool
    ├── reset(session_id)
    └── cleanup_expired()
```

### DreamZero Pipeline (PR4)

```
DreamZeroPipeline(nn.Module, CFGParallelMixin, SupportImageInput, ProgressBarMixin)
│
│  Components (loaded in __init__):
│  ├── text_encoder: UMT5EncoderModel          # from_pretrained (Wan2.1 base)
│  ├── image_encoder: CLIPVisionModel           # from_pretrained (Wan2.1 base)
│  ├── vae: DistributedAutoencoderKLWan         # from_pretrained (Wan2.1 base)
│  ├── transformer: CausalWanModel              # structure only, weights via loader
│  └── scheduler: VideoActionScheduler          # wraps video + action schedulers
│
│  Overrides:
│  ├── predict_noise(**kwargs) → (video_pred, action_pred)
│  ├── diffuse(...)            # prefill + denoising loop
│  └── load_weights(weights)   # key remapping: action_head.* → *
│
│  weights_sources: [ComponentSource(DreamZero checkpoint)]
│
CausalWanModel(nn.Module)
│  ├── patch_embedding: Conv3d
│  ├── text_embedding: Sequential
│  ├── time_embedding: Sequential
│  ├── time_projection: Sequential
│  ├── blocks: ModuleList[CausalWanAttentionBlock × 40]
│  ├── head: CausalHead
│  ├── img_emb: MLPProj
│  ├── action_encoder: MultiEmbodimentActionEncoder
│  ├── state_encoder: CategorySpecificMLP
│  ├── action_decoder: CategorySpecificMLP
│  ├── freqs, freqs_action, freqs_state   # RoPE buffers
│  │
│  │  GPU state (persists across calls):
│  ├── kv_cache_cond: list[Tensor]
│  ├── kv_cache_uncond: list[Tensor]
│  ├── current_start_frame: int
│  ├── clip_feas: Tensor
│  ├── ys: Tensor
│  └── language: Tensor
│
VideoActionScheduler
│  ├── video_scheduler: FlowUniPCMultistepScheduler
│  ├── action_scheduler: FlowUniPCMultistepScheduler
│  └── step(pred, t, latents) → tuple   # dispatches to both
```

### Serving Layer (PR5)

```
api_server.py
│  @router.websocket("/v1/world/roboarena")
│  └── → handler.handle_session(websocket)
│
OmniWorldStreamHandler (serving_world_stream.py)
│  ├── _engine: DiffusionEngine
│  ├── _session_store: WorldSessionStore
│  ├── _server_config: PolicyServerConfig
│  │
│  ├── handle_session(websocket)      # main loop
│  ├── _decode_obs(data) → dict       # msgpack decode + key mapping
│  ├── _encode_action(result) → bytes # (N,8) numpy → msgpack
│  ├── _build_request(obs, session) → OmniDiffusionRequest
│  ├── _update_session(session, result)
│  └── _reset_session(obs)
│
OmniServingWorld (serving_world.py)
│  ├── _engine: DiffusionEngine
│  ├── predict(request) → result      # calls engine.step
│  ├── create_session(request)
│  └── destroy_session(session_id)
```

### Data Flow (per infer call)

```
Client (test_client_AR)
  │ msgpack({obs, session_id, endpoint="infer"})
  ▼
OmniWorldStreamHandler
  │ decode obs, key mapping
  │ session = store.get_or_create(session_id)
  │ pack session state → extra_args
  ▼
DiffusionEngine.step(request)
  ├── pre_process_func: frame accumulate + select from extra_args
  ├── pipeline.forward: prefill → 4-step denoise → (video, action)
  └── post_process_func: passthrough
  │
  ▼ DiffusionOutput(custom_output={session_id, actions, current_start_frame})
OmniWorldStreamHandler
  │ update session state from custom_output
  │ convert action → (N,8) numpy
  │ msgpack encode
  ▼
Client
```
