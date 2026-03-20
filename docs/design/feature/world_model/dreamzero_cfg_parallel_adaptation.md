# DreamZero CFG Parallel Adaptation Plan

Tracking document for adapting DreamZero's CFG parallel inference into vllm-omni's `CFGParallelMixin`.

---

## Background

DreamZero uses a custom P2P-based CFG parallel mechanism (`ip_rank`/`ip_size`/`ip_group`) in `WANPolicyHead`. vllm-omni provides `CFGParallelMixin` with `all_gather`-based CFG parallel. This document tracks the design decisions for bridging the two.

**Reference files:**
- DreamZero: `~/code/dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- DreamZero server: `~/code/dreamzero/socket_test_optimized_AR.py`
- vllm-omni mixin: `vllm_omni/diffusion/distributed/cfg_parallel.py`
- vllm-omni design doc: `docs/design/feature/cfg_parallel.md`

---

## Current DreamZero CFG Parallel Summary

| Component | DreamZero Implementation |
|-----------|------------------------|
| Init | `parallelize(device_mesh)` → `ip_rank`/`ip_size` from DeviceMesh `["ip"]` |
| KV cache split | `_get_caches()`: rank 0 → `kv_cache1` (cond), rank 1 → `kv_cache_neg` (uncond) |
| Text prompt split | `_prepare_text_inputs()`: rank 0 → `text`, rank 1 → `text_negative` |
| Model forward | `_run_diffusion_steps()` loops over `context` list: 1 GPU runs 2 iters, 2 GPUs each run 1 |
| Result exchange | `_exchange_predictions()`: P2P `isend`/`irecv` of `(video_pred, action_pred)` |
| CFG combine | Caller: `flow_pred = uncond + cfg_scale * (cond - uncond)` |
| Scheduler step | Two independent schedulers (video + action), no broadcast sync |

## vllm-omni `CFGParallelMixin` Summary

| Component | vllm-omni Implementation |
|-----------|------------------------|
| Init | `get_classifier_free_guidance_world_size()` / `get_cfg_group()` from parallel_state |
| Model forward | `predict_noise_maybe_with_cfg()`: rank 0 → `predict_noise(**positive_kwargs)`, rank 1 → `predict_noise(**negative_kwargs)` |
| Result exchange | `cfg_group.all_gather()` (collective, not P2P) |
| CFG combine | `combine_cfg_noise()`: `uncond + scale * (cond - uncond)` on rank 0 |
| Scheduler step | `scheduler_step_maybe_with_cfg()`: rank 0 steps + `broadcast` syncs latents |

---

## Denoising Loop Data Flow

Understanding how data flows through each denoising step is critical for all diffs.

```
Step N:
                    ┌──────────────────────────────────────┐
                    │  同一份 noisy_input (video latent)     │
                    │  同一份 noisy_input_action (action)    │
                    └────────┬─────────────┬───────────────┘
                             │             │
                     ┌───────▼──────┐ ┌────▼────────────┐
                     │  cond 分支    │ │  uncond 分支      │
                     │  context[0]  │ │  context[1]      │
                     │  kv_cache[0] │ │  kv_cache[1]     │
                     │  (rank 0)    │ │  (rank 1)        │
                     └───────┬──────┘ └────┬────────────┘
                             │             │
                     flow_pred_cond   flow_pred_uncond
                     flow_pred_cond_  flow_pred_uncond_
                       action           action (丢弃)
                             │             │
                             ▼             ▼
              video_pred = uncond + 5.0 * (cond - uncond)   ← CFG combine
              action_pred = flow_pred_cond_action            ← 只取 cond
                             │             │
                     ┌───────▼──────┐ ┌────▼────────────┐
                     │ video sched  │ │ action sched     │
                     │   .step()    │ │   .step()        │
                     └───────┬──────┘ └────┬────────────┘
                             │             │
                     noisy_input      noisy_input_action
                     (updated)        (updated)
                             │             │
                             ▼             ▼
                    broadcast to both ranks (for CFG parallel)
                             │
Step N+1:           同一份传给 cond 和 uncond 两个分支
```

**Key observations:**
1. **Input symmetry:** Both cond and uncond branches receive the **same** `noisy_input` and `noisy_input_action`. The only difference is `context` (prompt embedding) and `kv_cache`.
2. **Output asymmetry:** Video uses CFG combine (`uncond + scale*(cond - uncond)`). Action only uses the cond prediction; the uncond action prediction is discarded.
3. **Sync requirement:** After scheduler step, updated `noisy_input` and `noisy_input_action` must be broadcast to all ranks so the next iteration's input is identical across ranks.
4. **Why action skips CFG:** CFG amplifies deviation from unconditional prediction to match text guidance. For robot actions (precise joint angles), this amplification causes instability. Actions only need the conditional prediction.

---

## Differences & TODO

Each difference requires discussion before implementation. Work through them one by one.

### Diff 1: Dual Output (video + action)

- **Status:** DECIDED
- **Problem:** `CFGParallelMixin.predict_noise()` assumes a single tensor return. DreamZero returns `(video_noise_pred, action_noise_pred)` — two tensors. Action only uses positive (conditional) prediction; uncond action prediction is discarded.
- **Decision:** Extend mixin to support tuple output. Add parameter `cfg_combine_mask` to `predict_noise_maybe_with_cfg()`.
- **API:**
  ```python
  cfg_combine_mask: tuple[str, ...] | None = None
  # Values per element: "cfg" | "positive" | "negative"
  #   "cfg"      → apply CFG combine: negative + scale * (positive - negative)
  #   "positive" → take positive (rank 0) prediction only
  #   "negative" → take negative (rank 1) prediction only
  ```
- **Behavior:**
  - `None` (default) → existing behavior: single tensor, apply CFG. Fully backward-compatible.
  - When set, `predict_noise()` must return a tuple. Each element is all_gathered independently. Combine logic is applied per-element according to the mask.
- **DreamZero usage:**
  ```python
  video_pred, action_pred = self.predict_noise_maybe_with_cfg(
      ...,
      cfg_combine_mask=("cfg", "positive"),
  )
  ```
- **Naming rationale:** Uses `"positive"` / `"negative"` to align with mixin's existing `positive_kwargs` / `negative_kwargs` naming, rather than `"cond"` / `"uncond"` from diffusion literature.

### Diff 2: Dual Scheduler (video + action)

- **Status:** DECIDED
- **Prerequisite PR:** "Reduce redundant broadcast in CFG parallel" (`PR1_reduce_redundant_broadcast_cfg_parallel.md`). This PR makes all ranks compute combine + scheduler step locally after all_gather, eliminating the broadcast entirely.
- **Problem:** `scheduler_step_maybe_with_cfg()` assumes one scheduler and one latent tensor. DreamZero has two independent scheduler flows: (1) video scheduler with optional sigma rescaling for decoupled inference, (2) action scheduler with standard schedule. Both ranks use identical scheduler configs (no cross-rank divergence).
- **Decision:** DreamZero implements a `VideoActionScheduler` wrapper that packs two schedulers behind a single `.step()` interface accepting/returning tuples. After the prerequisite PR, `scheduler_step_maybe_with_cfg` simply calls `self.scheduler_step` on all ranks — no broadcast needed, so tuple support comes for free (each rank locally steps both schedulers).
- **DreamZero usage:**
  ```python
  self.scheduler = VideoActionScheduler(video_scheduler, action_scheduler)
  # After prerequisite PR: all ranks compute scheduler_step locally
  # Tuple in, tuple out, no broadcast needed
  ```

### Diff 3: Stateful KV Cache (cross-call persistence)

- **Status:** DECIDED
- **Problem:** `CFGParallelMixin` is stateless — each timestep's kwargs contain all info. DreamZero's KV cache persists across AR calls and across denoising steps. Cond and uncond have **separate** KV caches (`kv_cache1`/`kv_cache_neg`).
- **Decision:** Option A — manage KV caches in the pipeline, not the mixin. Mixin stays stateless.
  - Each rank holds only its own cache. In CFG parallel mode: rank 0 holds cond cache, rank 1 holds uncond cache.
  - KV cache is passed via `positive_kwargs` / `negative_kwargs`:
    ```python
    positive_kwargs = {"kv_cache": self.kv_cache_cond, ...}   # rank 0 uses
    negative_kwargs = {"kv_cache": self.kv_cache_uncond, ...}  # rank 1 uses
    ```
  - After the "reduce redundant broadcast" PR, both ranks run `predict_noise` locally. Each rank only ever receives its own kwargs, so each rank only touches its own cache. No cache synchronization needed.
  - Cross-attention caches follow the same pattern.
  - In single-GPU sequential mode, pipeline holds both caches and passes them in two sequential calls.
- **Note:** KV cache shape is `[2, B, seq, heads, dim]` per layer × 40 layers. ~2GB per session with CFG.

### Diff 4: Prefill Phase (KV cache initialization)

- **Status:** DECIDED
- **Problem:** DreamZero has two prefill calls before the denoising loop: (1) first-frame encoding at `current_start_frame == 0`, (2) new observation encoding at `current_start_frame != 1`. Both run the model with `t=0`, `action=None`, `update_kv_cache=True`, and discard the return value.
- **Decision:** Handled entirely in `diffuse()`, before the `for t in timesteps` loop. Prefill reuses `predict_noise_maybe_with_cfg` as a convenience to run both ranks' forwards — return value is discarded. Mixin unchanged.
  ```python
  def diffuse(self, ...):
      # Prefill 1: first frame (session start)
      if self.current_start_frame == 0:
          _ = self.predict_noise_maybe_with_cfg(
              do_true_cfg=True, ...,
              positive_kwargs={..., "update_kv_cache": True},
              negative_kwargs={..., "update_kv_cache": True},
          )
          self.current_start_frame += 1

      # Prefill 2: new observation frames (AR steps)
      if self.current_start_frame != 1:
          _ = self.predict_noise_maybe_with_cfg(...)

      # Denoising loop
      for t in timesteps:
          ...
  ```

### Diff 5: DiT Cache Acceleration (skip steps)

- **Status:** DECIDED
- **Prerequisite PR:** "StepCache: Step-Level Prediction Caching" (`PR2_step_cache.md`). Adds step-level caching to `CFGParallelMixin.predict_noise_maybe_with_cfg` — all models get this for free.
- **Problem:** DreamZero's `should_run_model()` skips entire DiT forward when consecutive predictions are similar (cosine similarity > 0.95). This is critical for 7Hz real-time inference.
- **Decision:** DreamZero uses StepCache directly, no custom skip logic needed.
  ```python
  self._step_cache = StepCache(
      similarity_fn=torch.nn.functional.cosine_similarity,
      threshold=0.95,
      warmup_steps=2,
      max_skip_steps=4,  # DreamZero's countdown mechanism
  )
  ```
- **CFG parallel coordination:** After "reduce redundant broadcast" PR, both ranks compute the same combined prediction locally → both ranks make the same skip decision → no broadcast needed for skip coordination.
- **Composable:** StepCache (step-level) stacks with TeaCache (transformer-level) and cache-dit (block-level). DreamZero can use StepCache alone or combined with other accelerators.

### Diff 6: Communication Pattern (P2P vs all_gather)

- **Status:** DECIDED
- **Problem:** DreamZero uses P2P `isend`/`irecv`. vllm-omni uses `all_gather`. For 2-rank CFG, performance difference is negligible.
- **Decision:** Use vllm-omni's `all_gather`. No adaptation needed. After "reduce redundant broadcast" PR, `all_gather` is the only communication per step — simpler than DreamZero's original P2P + no broadcast.

---

## Final Approach

### Prerequisite PRs (model-agnostic, benefit all pipelines)

1. **"Reduce redundant broadcast in CFG parallel"** (`PR1_reduce_redundant_broadcast_cfg_parallel.md`)
   - All ranks compute combine + scheduler step locally after all_gather
   - Removes broadcast, simplifies mixin, improves extensibility

2. **"StepCache: Step-level prediction caching"** (`PR2_step_cache.md`)
   - Add `_step_cache` support to `predict_noise_maybe_with_cfg`
   - Skip entire denoising steps when predictions are stable
   - Zero changes to existing models

3. **"cfg_combine_mask for tuple output"** (`PR1_reduce_redundant_broadcast_cfg_parallel.md`)
   - Extend `predict_noise_maybe_with_cfg` to support tuple returns with per-element combine mode
   - New param `cfg_combine_mask=("cfg", "positive", ...)`

### DreamZero-specific implementation

```python
class DreamZeroPipeline(nn.Module, CFGParallelMixin, ...):

    def __init__(self, ...):
        # Dual scheduler wrapped as one
        self.scheduler = VideoActionScheduler(video_scheduler, action_scheduler)
        # Step cache (DreamZero's should_run_model)
        self._step_cache = StepCache(
            similarity_fn=cosine_similarity, threshold=0.95,
            warmup_steps=2, max_skip_steps=4,
        )

    def predict_noise(self, **kwargs):
        video_pred, action_pred, _ = self.transformer(**kwargs)
        return (video_pred, action_pred)

    def diffuse(self, ...):
        # Prefill (before loop)
        if self.current_start_frame == 0:
            _ = self.predict_noise_maybe_with_cfg(
                ..., positive_kwargs={..., "kv_cache": self.kv_cache_cond},
                negative_kwargs={..., "kv_cache": self.kv_cache_uncond},
            )
        # Denoising loop
        for t in timesteps:
            video_pred, action_pred = self.predict_noise_maybe_with_cfg(
                ..., cfg_combine_mask=("cfg", "positive"),
                positive_kwargs={..., "kv_cache": self.kv_cache_cond},
                negative_kwargs={..., "kv_cache": self.kv_cache_uncond},
            )
            noisy_video, noisy_action = self.scheduler_step_maybe_with_cfg(
                (video_pred, action_pred), (video_t, action_t),
                (noisy_video, noisy_action), do_true_cfg,
            )
```

---

## Discussion Log

Record decisions here as each diff is discussed.

| Diff | Date | Decision | Notes |
|------|------|----------|-------|
| 1 | 2026-03-20 | Option B: extend mixin with `cfg_combine_mask` param | Tuple support + per-element `"cfg"`/`"positive"`/`"negative"` mode. Backward-compatible (None = old behavior). |
| 2 | 2026-03-20 | `VideoActionScheduler` wrapper; prerequisite PR removes broadcast entirely | After "reduce redundant broadcast" PR, all ranks compute locally. Tuple support is free — no broadcast, no mask. |
| 3 | 2026-03-20 | Pipeline manages caches, mixin stays stateless | KV cache in kwargs. Each rank holds its own cache. No sync needed. |
| 4 | 2026-03-20 | Pipeline handles in `diffuse()` before loop, mixin unchanged | Reuses `predict_noise_maybe_with_cfg` as convenience, discards return. |
| 5 | 2026-03-20 | Use StepCache from prerequisite PR | DreamZero's `should_run_model` maps directly to StepCache config. No custom logic needed. |
| 6 | 2026-03-20 | Use vllm-omni's `all_gather`, drop P2P | Simpler, generalizes to >2 ranks, negligible perf diff at 2 ranks. |
