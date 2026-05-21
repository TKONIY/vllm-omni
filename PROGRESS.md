# UAD Integration Progress

## 2026-05-20: EngineCore integration scaffold

Completed:

- Added the top-level `uad_vllm` package for the new EngineCore-shaped UAD design.
- Defined initial UAD request state, scheduler items, runner outputs, executor facade, runner facade, and worker/model-runner placeholders.
- Implemented `UADEngineCore.step()` orchestration so the UAD path already calls scheduler, executor, runner output processing, and scheduler update in the intended order.
- Added a `--uad-engine` serve flag and `VLLM_OMNI_USE_UAD_ENGINE=1` process switch.
- Wired vLLM-Omni stage startup so the UAD switch resolves `StageEngineCoreProc` to `UADEngineCore`.
- Kept current AR behavior as passthrough; DiT/artifact execution is left as explicit TODO interfaces.
- Added focused unit tests for config selection, scheduler passthrough, and `UADEngineCore.step()` call order.

Validation:

- `uv run --no-sync ruff check uad_vllm tests/uad_vllm vllm_omni/entrypoints/cli/serve.py vllm_omni/engine/stage_engine_core_proc.py vllm_omni/engine/stage_engine_startup.py`
- `uv run --no-sync python -m compileall -q uad_vllm tests/uad_vllm vllm_omni/entrypoints/cli/serve.py vllm_omni/engine/stage_engine_core_proc.py vllm_omni/engine/stage_engine_startup.py`
- `git diff --check -- uad_vllm tests/uad_vllm vllm_omni/entrypoints/cli/serve.py vllm_omni/engine/stage_engine_core_proc.py vllm_omni/engine/stage_engine_startup.py pyproject.toml`
- `uv run --no-sync python -m pytest tests/uad_vllm -q` (`7 passed`)

Next step:

- Implement the first real UAD scheduler state transition and request attachment path.

## 2026-05-21: v1 SchedulerInterface compatibility fix

Completed:

- Changed UAD abstractions to inherit the relevant v1 interfaces where they occupy v1-style slots.
- Made `UADScheduler` inherit `SchedulerInterface` and explicitly delegate the full scheduler interface to the base scheduler.
- Made `UADSchedulerOutput` inherit `SchedulerOutput` and carry additive `uad_items` metadata instead of wrapping `base_output`.
- Made `UADExecutor` inherit `Executor` while delegating to the base executor instance.
- Made `UADGPUWorker` inherit `WorkerBase`.
- Added `UADModelRunnerOutput(ModelRunnerOutput)` for additive UAD phase outputs.
- Kept `EngineCoreProc.scheduler` pointing at the original v1 scheduler; `self.uad_scheduler` is used only by `UADEngineCore.step()`.

Validation:

- `uv run --no-sync ruff check uad_vllm tests/uad_vllm vllm_omni/entrypoints/cli/serve.py vllm_omni/engine/stage_engine_core_proc.py vllm_omni/engine/stage_engine_startup.py`
- `uv run --no-sync python -m compileall -q uad_vllm tests/uad_vllm vllm_omni/entrypoints/cli/serve.py vllm_omni/engine/stage_engine_core_proc.py vllm_omni/engine/stage_engine_startup.py`
- `uv run --no-sync python -m pytest tests/uad_vllm -q` (`8 passed`)

## 2026-05-21: Design document sync

Completed:

- Added `docs/uad/design_uad.md` to the `uad-integration` branch.
- Rewrote the design around the current vLLM v1-compatible implementation.
- Removed the old nano-UAD toy scheduler framing from the active design.
- Documented the current inheritance/composition contract for `UADScheduler`, `UADSchedulerOutput`, `UADExecutor`, `UADModelRunnerOutput`, and `UADGPUWorker`.
- Documented current TODO boundaries for request state attachment, DiT scheduling/execution, KV commit, CFG parallel, and SP.
- Restored detailed state ownership, UAD request fields, phase update fields, and typical update timing under the current v1-compatible design.
- Added explicit schedule-time persistence metadata via `UADScheduleItem.num_persistent_tokens` and documented its relationship to update-time `UADPhaseUpdate.num_new_computed_tokens`.

Validation:

- `git diff --check -- docs/uad/design_uad.md PROGRESS.md`
- `uv run --no-sync ruff check uad_vllm tests/uad_vllm`
- `uv run --no-sync python -m compileall -q uad_vllm tests/uad_vllm`
- `uv run --no-sync python -m pytest tests/uad_vllm -q` (`9 passed`)

## 2026-05-21: Minimal inherited facade cleanup

Completed:

- Kept `UADScheduler(SchedulerInterface)` and `UADExecutor(Executor)` inheritance.
- Removed broad v1 lifecycle delegation from UAD facades.
- Kept only the UAD step path delegates: schedule, grammar, scheduler update, execute, and sample.
- Marked unsupported v1 lifecycle methods as explicit `NotImplementedError` stubs because `EngineCoreProc.scheduler` and `EngineCoreProc.model_executor` remain the real v1 lifecycle owners.
- Updated the design document and tests to reflect the narrower facade contract.

Validation:

- `uv run --no-sync ruff check uad_vllm tests/uad_vllm`
- `uv run --no-sync python -m compileall -q uad_vllm tests/uad_vllm`
- `uv run --no-sync python -m pytest tests/uad_vllm -q` (`10 passed`)
