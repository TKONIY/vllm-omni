# StepCache: Step-Level Prediction Caching for Diffusion Models

**PR Scope:** Add step-level caching to `CFGParallelMixin` that skips entire denoising steps when consecutive predictions are similar. Complements TeaCache (transformer-level) and cache-dit (block-level).

---

## Goals

1. **Reduce compute:** Skip entire `predict_noise` calls (including all transformer blocks + preprocessing) when predictions are stable across consecutive steps.
2. **Zero model changes:** Hook into `predict_noise_maybe_with_cfg` — the universal entry point all models already use. No changes to any existing pipeline's `diffuse()` loop.
3. **Composable with existing caching:** StepCache (step-level) + TeaCache (transformer-level) + cache-dit (block-level) can stack — StepCache decides first, if not skipped, TeaCache/cache-dit decide within the transformer.

---

## Caching Hierarchy

```
Denoising loop
│
├─ StepCache: skip entire step?              ← NEW (this PR)
│   │
│   └─ predict_noise_maybe_with_cfg
│       │
│       └─ predict_noise → transformer.forward
│           │
│           ├─ TeaCache: skip all blocks?     ← existing (residual reuse)
│           │
│           └─ cache-dit: skip some blocks?   ← existing (DBCache)
```

| Level | What it skips | Judgment signal | Skip mechanism |
|-------|--------------|-----------------|----------------|
| **StepCache** | Entire `predict_noise_maybe_with_cfg` call | Similarity of consecutive **predictions** (output) | Reuse previous prediction directly |
| **TeaCache** | All transformer blocks (but still does preprocess + postprocess) | L1 distance of **modulated input** (first block input) | `hidden_states + previous_residual` |
| **cache-dit** | Selected transformer blocks | Per-block residual diff | Reuse per-block residual |

---

## Hook Point

`predict_noise_maybe_with_cfg` is the universal entry point for all denoising steps in all models:

```python
# Every model's diffuse() loop:
for t in timesteps:
    noise_pred = self.predict_noise_maybe_with_cfg(...)  ← hook here
    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, ...)
```

All models go through this method regardless of:
- CFG enabled or not (no CFG → enters `else` branch, still returns a prediction)
- CFG parallel or sequential
- Single or dual transformer

StepCache intercepts at the **entry and exit** of this method:
- **Entry:** Check if we should skip → if yes, return cached prediction
- **Exit:** Cache the current prediction for next step's comparison

---

## Design

### StepCache class

```python
class StepCache:
    """Step-level prediction cache.

    Decides whether to skip a denoising step by comparing the current
    prediction to the previous one using a configurable similarity function.
    """

    def __init__(
        self,
        similarity_fn: Callable[[torch.Tensor, torch.Tensor], float],
        threshold: float,
        warmup_steps: int = 2,
        max_skip_steps: int = 1,
    ):
        self.similarity_fn = similarity_fn  # e.g., cosine_similarity
        self.threshold = threshold           # e.g., 0.95
        self.warmup_steps = warmup_steps     # always compute first N steps
        self.max_skip_steps = max_skip_steps # max consecutive skips
        self._cache: torch.Tensor | tuple | None = None
        self._step_count = 0
        self._consecutive_skips = 0

    def should_skip(self) -> bool:
        if self._cache is None:
            return False
        if self._step_count < self.warmup_steps:
            return False
        if self._consecutive_skips >= self.max_skip_steps:
            return False  # force compute after max_skip_steps
        return True  # tentative — actual decision after similarity check

    def check_and_maybe_skip(self, current_pred) -> tuple[bool, Any]:
        """Compare current pred with cache. Return (should_skip, cached_or_current)."""
        if not self.should_skip():
            self._record(current_pred)
            return False, current_pred

        sim = self._compute_similarity(current_pred, self._cache)
        if sim > self.threshold:
            self._consecutive_skips += 1
            self._step_count += 1
            return True, self._cache
        else:
            self._record(current_pred)
            return False, current_pred

    def get_cached(self):
        return self._cache

    def _record(self, pred):
        self._cache = pred
        self._step_count += 1
        self._consecutive_skips = 0

    def _compute_similarity(self, a, b):
        """Handle both Tensor and tuple of Tensors."""
        if isinstance(a, tuple):
            # Use first element (video pred) for similarity
            a, b = a[0], b[0]
        return self.similarity_fn(
            a.flatten(1).float(), b.flatten(1).float()
        ).mean().item()

    def reset(self):
        self._cache = None
        self._step_count = 0
        self._consecutive_skips = 0
```

### Integration into `predict_noise_maybe_with_cfg`

```python
class CFGParallelMixin(metaclass=ABCMeta):

    def predict_noise_maybe_with_cfg(self, ...):
        # ---- StepCache: early exit if skippable ----
        step_cache = getattr(self, '_step_cache', None)
        if step_cache is not None and step_cache.should_skip():
            return step_cache.get_cached()

        # ---- Existing logic (unchanged) ----
        if do_true_cfg:
            ...
        else:
            ...
        noise_pred = ...

        # ---- StepCache: record prediction ----
        if step_cache is not None:
            step_cache._record(noise_pred)

        return noise_pred
```

### Enabling StepCache in a pipeline

```python
class SomePipeline(nn.Module, CFGParallelMixin, ...):
    def __init__(self, ...):
        ...
        # Optional: enable step cache
        self._step_cache = None  # disabled by default

    def forward(self, ...):
        if use_step_cache:
            self._step_cache = StepCache(
                similarity_fn=torch.nn.functional.cosine_similarity,
                threshold=0.95,
                warmup_steps=2,
                max_skip_steps=4,
            )
        ...
        result = self.diffuse(...)
        ...
        if self._step_cache:
            self._step_cache.reset()
```

---

## Backward Compatibility

- `_step_cache` defaults to `None` on all pipelines (via `getattr` fallback)
- When `None`, the two added lines are two `None` checks — effectively zero overhead
- No signature changes to `predict_noise_maybe_with_cfg`
- No changes to any existing pipeline code

---

## Tuple Output Support

StepCache naturally handles tuple predictions (e.g., DreamZero's `(video_pred, action_pred)`):
- **Cache:** stores the entire tuple as-is
- **Similarity:** computed on the first element (primary output) by default
- **Return:** returns the cached tuple — caller unpacks as usual

---

## Interaction with other caching

| Scenario | Behavior |
|----------|----------|
| StepCache only | Skip entire step, reuse prev prediction |
| TeaCache only | Skip transformer blocks inside step, residual reuse |
| StepCache + TeaCache | StepCache decides first. If not skipped, TeaCache decides inside transformer |
| StepCache + cache-dit | Same — StepCache first, cache-dit inside if not skipped |
| All three | StepCache → TeaCache → cache-dit (cascading) |

No conflicts — each level is independent.

---

## Relationship to DreamZero

DreamZero's `should_run_model()` is a specific instance of StepCache with:
- `similarity_fn = cosine_similarity`
- `threshold = 0.95` (also 0.93 for 2-step skip)
- `warmup_steps = 2`
- `max_skip_steps = 4` (countdown mechanism)

After this PR, DreamZero uses StepCache directly instead of custom skip logic.
