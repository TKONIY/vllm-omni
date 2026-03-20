# Reduce Redundant Broadcast in CFG Parallel

**PR Scope:** Simplify `CFGParallelMixin` by letting all ranks compute CFG combine + scheduler step locally after `all_gather`.

---

## Goals

1. **Reduce communication:** Remove the redundant `broadcast` in `scheduler_step_maybe_with_cfg`. Currently every denoising step does `all_gather` + `broadcast` — the broadcast is unnecessary since all ranks already have the data after all_gather.
2. **Reduce code complexity:** Eliminate the rank-0-only combine logic and the `return None` path on rank 1. After the change, all ranks execute the same code — no branching by rank, no None checks.
3. **Improve extensibility:** With all ranks holding valid results, downstream code (scheduler step, tuple outputs, custom post-processing) doesn't need to be rank-aware. This unblocks features like dual-output models (DreamZero) and composite schedulers without adding per-element broadcast logic.

---

## Problem

Current `CFGParallelMixin` flow:

```
rank 0: predict_noise(**positive)  ──┐
rank 1: predict_noise(**negative)  ──┤
                                     │
                              all_gather          ← communication 1
                                     │
rank 0: combine + scheduler_step → latents
rank 1: return None (idle)
                                     │
                              broadcast(latents)  ← communication 2 (redundant)
                                     │
rank 0: latents (updated)
rank 1: latents (received)
```

After `all_gather`, **both ranks already have identical copies of positive and negative predictions**. The combine formula (`neg + scale * (pos - neg)`) and scheduler step are both deterministic pure math — no randomness, no rank-dependent state. So rank 1 can compute the exact same result locally. The `broadcast` is redundant.

Additional problems with the current design:
- **Rank 1 is idle** after all_gather — wastes compute capacity
- **`predict_noise_maybe_with_cfg` returns `None` on rank 1** — forces all callers to handle the None case or implicitly ignore it
- **`scheduler_step_maybe_with_cfg` has rank-branching logic** — rank 0 steps, rank 1 waits, then broadcast. This complexity compounds when extending to tuple outputs (need per-element broadcast, broadcast masks, etc.)

## Proposed Change

```
rank 0: predict_noise(**positive)  ──┐
rank 1: predict_noise(**negative)  ──┤
                                     │
                              all_gather          ← communication 1 (only)
                                     │
rank 0: combine + scheduler_step → latents
rank 1: combine + scheduler_step → latents  (same result, computed locally)
```

One `all_gather` instead of `all_gather` + `broadcast`. For DreamZero with tuple outputs this saves **2 extra broadcasts** (video + action) per denoising step × 4 steps = 8 fewer communications per chunk.

## Implementation

### Change 1: `predict_noise_maybe_with_cfg` — all ranks compute combine

```python
def predict_noise_maybe_with_cfg(self, ...):
    if do_true_cfg:
        cfg_parallel_ready = get_classifier_free_guidance_world_size() > 1

        if cfg_parallel_ready:
            cfg_group = get_cfg_group()
            cfg_rank = get_classifier_free_guidance_rank()

            if cfg_rank == 0:
                local_pred = self.predict_noise(**positive_kwargs)
            else:
                local_pred = self.predict_noise(**negative_kwargs)

            if output_slice is not None:
                local_pred = local_pred[:, :output_slice]

            gathered = cfg_group.all_gather(local_pred, separate_tensors=True)

-           if cfg_rank == 0:
-               noise_pred = gathered[0]
-               neg_noise_pred = gathered[1]
-               noise_pred = self.combine_cfg_noise(...)
-               return noise_pred
-           else:
-               return None
+           # All ranks compute combine (deterministic, same result)
+           noise_pred = gathered[0]       # positive
+           neg_noise_pred = gathered[1]   # negative
+           noise_pred = self.combine_cfg_noise(
+               noise_pred, neg_noise_pred, true_cfg_scale, cfg_normalize
+           )
+           return noise_pred

        else:
            # Sequential: unchanged
            ...
    else:
        # No CFG: unchanged
        ...
```

### Change 2: `scheduler_step_maybe_with_cfg` — remove broadcast

```python
def scheduler_step_maybe_with_cfg(self, noise_pred, t, latents, do_true_cfg):
-   cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
-
-   if cfg_parallel_ready:
-       cfg_group = get_cfg_group()
-       cfg_rank = get_classifier_free_guidance_rank()
-
-       if cfg_rank == 0:
-           latents = self.scheduler_step(noise_pred, t, latents)
-
-       latents = latents.contiguous()
-       cfg_group.broadcast(latents, src=0)
-   else:
-       latents = self.scheduler_step(noise_pred, t, latents)
-
-   return latents
+   # All ranks now have valid noise_pred (after all_gather + local combine).
+   # Scheduler step is deterministic, so all ranks compute locally.
+   # No broadcast needed.
+   latents = self.scheduler_step(noise_pred, t, latents)
+   return latents
```

After this change, `scheduler_step_maybe_with_cfg` becomes equivalent to `scheduler_step`. It can be kept for backward compatibility (callers don't need to change), or deprecated in a follow-up.

## Correctness Argument

For this optimization to be correct, the following must hold:

1. **`all_gather` gives identical data to all ranks** — guaranteed by NCCL semantics
2. **`combine_cfg_noise` is deterministic** — `neg + scale * (pos - neg)` is pure arithmetic, same inputs → same output
3. **`cfg_normalize_function` is deterministic** — `torch.norm` + division, deterministic on same input
4. **`scheduler.step` is deterministic** — all schedulers in vllm-omni (FlowUniPC, FlowMatchEuler, etc.) are deterministic given same inputs. No random sampling in the scheduler step.
5. **Floating point reproducibility** — both ranks run on the same GPU architecture, same dtype. NCCL `all_gather` delivers bit-identical tensors. The subsequent compute is the same ops on the same data → bit-identical results.

**Risk:** If a future scheduler step uses randomness (e.g., stochastic samplers), this optimization breaks. Mitigation: document the determinism requirement in the mixin docstring.

## Backward Compatibility

- **Callers:** Zero change needed. `predict_noise_maybe_with_cfg` still returns `Tensor`. `scheduler_step_maybe_with_cfg` still returns `Tensor`. Same signatures.
- **Behavior change:** `predict_noise_maybe_with_cfg` now returns a valid tensor on **all ranks** (previously `None` on rank 1). Callers that check `if noise_pred is None` (rank 1 skip pattern) need review.

### Callers that check return value:

```bash
grep -rn "predict_noise_maybe_with_cfg" vllm_omni/diffusion/models/ | grep -i "none\|rank\|if.*pred"
```

Need to verify no caller on rank 1 does `if noise_pred is None: skip`. If they do, they'll now execute the scheduler step too — which is correct since we want all ranks to step.

## Testing

1. Run existing CFG parallel tests — output should be **bit-identical** (no behavior change, just fewer communications)
2. Benchmark: measure wall time with and without the change on a 2-GPU CFG parallel run
3. Verify with `NCCL_DEBUG=INFO` that broadcast calls are eliminated

## Extension: `cfg_combine_mask` for Tuple Output

With all ranks computing combine locally, this PR also adds `cfg_combine_mask` to support models that return multiple predictions with different combine strategies (e.g., DreamZero returns video + action).

### New parameter

```python
def predict_noise_maybe_with_cfg(
    self, ...,
    cfg_combine_mask: tuple[str, ...] | None = None,  # NEW
):
```

### Semantics

- **Type:** `tuple[str, ...]`, each element is `"cfg"` | `"positive"` | `"negative"`
- **Default:** `None` → existing single-tensor behavior, fully backward-compatible
- **When set:** `predict_noise()` must return a tuple with matching length

| Mode | After all_gather |
|------|-----------------|
| `"cfg"` | `negative + scale * (positive - negative)` |
| `"positive"` | Take positive branch (rank 0) result |
| `"negative"` | Take negative branch (rank 1) result |

### Implementation (extends Change 1 above)

```python
if cfg_combine_mask is not None:
    # Tuple path
    local_preds = self.predict_noise(**kwargs)  # returns tuple
    gathered = [cfg_group.all_gather(p, separate_tensors=True) for p in local_preds]

    results = []
    for i, mode in enumerate(cfg_combine_mask):
        pos, neg = gathered[i][0], gathered[i][1]
        if mode == "cfg":
            results.append(self.combine_cfg_noise(pos, neg, true_cfg_scale, cfg_normalize))
        elif mode == "positive":
            results.append(pos)
        elif mode == "negative":
            results.append(neg)
    return tuple(results)
```

Sequential and no-CFG paths follow the same pattern.

### Backward compatibility

`cfg_combine_mask=None` (default) → enters existing single-tensor code path → zero change for all existing models.

---

## Relationship to DreamZero

This PR is a prerequisite for DreamZero adaptation:
- **Diff 1 (dual output):** `cfg_combine_mask=("cfg", "positive")` — video does CFG, action takes positive only
- **Diff 2 (dual scheduler):** With broadcast removed, `scheduler_step_maybe_with_cfg` just calls `self.scheduler_step` on all ranks — tuple returns from `VideoActionScheduler` work naturally
- **Diff 5 (step cache):** Both ranks compute identical combined predictions → identical skip decisions → no coordination needed
