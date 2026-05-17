# PROGRESS

## Step 0: Toy HunyuanImage3 UAD Entry

Status: completed.

Completed modifications:

- Added the minimal UAD package skeleton under `vllm_omni/uad/`.
- Added `UADEngine` / `AsyncUADEngine` with request add, scheduler step, runner execution, and output application.
- Added `UADRequestState`, `UADToken`, scheduler items, and output dataclasses for the Step 0 ledger.
- Added `HunyuanImage3UADAdapter` at `vllm_omni/uad/omni/adapter/hunyuan_image3.py`.
- Added toy `HunyuanImage3UADForConditionalGeneration` at `vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py`.
- Added Step 0 tests in `tests/uad/test_step0.py`.
- Updated `AGENTS.md` with the stop-for-review, validation, push, and progress-recording rule.

Validation:

- Passed: `ruff check vllm_omni/uad vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py tests/uad/test_step0.py`.
- Passed: `python -m compileall -q vllm_omni/uad vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py tests/uad/test_step0.py`.
- Passed: `git diff --check -- AGENTS.md PROGRESS.md vllm_omni/uad vllm_omni/model_executor/models/hunyuan_image3 tests/uad/test_step0.py`.
- Blocked: `pytest tests/uad/test_step0.py -q` in the current Python 3.13 environment because global pytest setup imports `torch`, which is not installed.

Commit and push:

- Local commit created for Step 0.
- Blocked: pushing to `fork/uad/dev` was rejected because the remote already has `refs/heads/uad`; Git cannot also create `refs/heads/uad/dev` without deleting or renaming the existing `uad` branch.
- Per user follow-up, Step 0 is pushed to the current branch upstream `fork/uad` instead.

## Step 1: UAD Scheduler Shadow Item

Status: completed.

Completed modifications:

- Extended `UADSchedulerOutput` with `base_output`, request id listing, total scheduled token count, and per-request scheduled token count.
- Added `num_computed_tokens` to `UADScheduleItem` so each shadow item records the request computed-prefix state at scheduling time.
- Added `UADShadowScheduler.build_shadow_output()` and kept `UADToyScheduler.schedule()` as the Step 1 toy scheduler entrypoint.
- Added Step 1 scheduler tests in `tests/uad/test_step1_scheduler.py` for prefill, decode, finished-request skipping, and scheduled-token totals.

Validation:

- Passed: `ruff check vllm_omni/uad tests/uad/test_step0.py tests/uad/test_step1_scheduler.py`.
- Passed: `python -m compileall -q vllm_omni/uad tests/uad/test_step0.py tests/uad/test_step1_scheduler.py`.
- Passed: `git diff --check -- PROGRESS.md vllm_omni/uad tests/uad/test_step0.py tests/uad/test_step1_scheduler.py`.
- Blocked: `pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py -q` in the current Python 3.13 environment because global pytest setup imports `torch`, which is not installed.

Commit and push:

- Step 1 is committed and pushed to the current branch upstream `fork/uad`.

## Step 2: HunyuanImage3 Toy Phase Switch

Status: completed.

Completed modifications:

- Added `UADPhaseUpdate` and DiT/image boundary fields to `UADRequestState`.
- Extended `UADModelOutput` so runner outputs can carry a request state-machine update.
- Updated `UADEngine` to apply phase updates after scheduled AR tokens are marked computed and new engine/materialized tokens are appended.
- Updated `UADShadowScheduler` to skip `dit_step` requests in Step 2, so pending image context tokens are not accidentally scheduled as AR decode tokens.
- Added `HunyuanImage3UADStateConfig` for HunyuanImage3 special-token rules: `<img_ratio_*>` detection, stage-transition helpers, ratio-index extraction, engine-only token filtering, and toy image-context token construction.
- Updated `HunyuanImage3UADAdapter` so a sampled ratio token appends ratio + toy image context tokens, switches the request to `dit_step`, records ratio metadata, and leaves those new tokens uncomputed until a later cache-commit step.
- Updated `docs/uad/plan_uad.md` Step 2 to explicitly map HunyuanImage3 tokenizer/sampler rules, ignored text-mask metadata, AR KV reuse metadata, and later-step obligations.
- Added Step 2 tests in `tests/uad/test_step2_phase_switch.py`.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py tests/uad`.
- Passed: `git diff --check -- PROGRESS.md docs/uad/plan_uad.md vllm_omni/uad vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3_uad.py tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py -q`.

Environment note:

- `uv run --extra dev` initially resolved `torch==2.12.0+cu130`, which was ABI-incompatible with the installed `vllm==0.20.0` extension. The local `.venv` was corrected with `uv pip install 'torch==2.11.0' 'torchaudio==2.11.0' 'torchvision==0.26.0'`, matching vLLM metadata, and validation was then run with `uv run --no-sync`.

Commit and push:

- Step 2 is committed and pushed to the current branch upstream `fork/uad`.

## Design Update: Runner-first UAD Path

Status: completed.

Completed modifications:

- Updated `docs/uad/design_uad.md` so the long-term UAD abstraction starts at `UADRunner`, not a separate model translation layer.
- Replaced `UADModelAdapter` references with runner-owned input building, model execution, output processing, state-machine updates, and HunyuanImage3 special-token helpers.
- Updated `docs/uad/plan_uad.md` so Step 3 first folds the Step 0/2 toy single-item execution responsibilities into `UADRunner`, then adds toy DiT step scheduling.
- Updated the plan status to show Step 2 as completed and the next implementation step as runner-first Step 3.

Validation:

- Passed: `if rg -n "UADModelAdapter|HunyuanImage3UADAdapter" docs/uad/design_uad.md docs/uad/plan_uad.md; then exit 1; else exit 0; fi`.
- Passed: `git diff --check -- PROGRESS.md docs/uad/design_uad.md docs/uad/plan_uad.md`.

Commit and push:

- Runner-first design update is committed and pushed to the current branch upstream `origin/uad`.

## Experiment Cleanup: Hunyuan Batch/Route Profiling

Status: completed.

Scope:

- Consolidate the existing HunyuanImage3 MoE batch-size/backend profiling scripts and the later real-request routing
  profiling helpers under `docs/uad/script/`.
- Keep generated traces, HTML reports, cloned official repositories, logs, and large local outputs under `artifacts/`
  as local-only data.
- Add the HunyuanImage3 MoE route tracer as opt-in instrumentation controlled by environment variables.

Validation:

- Passed: `uv run --no-sync ruff check docs/uad/script/build_hunyuan_real_request_routing_html.py docs/uad/script/build_hunyuan_routing_tabs_html.py docs/uad/script/profile_hunyuan_a13b_routing.py docs/uad/script/profile_hunyuan_official_clean_routing.py docs/uad/script/profile_hunyuan_official_routing.py docs/uad/script/profile_hunyuan_real_request_routing.py vllm_omni/model_executor/models/hunyuan_image3/moe_route_trace.py vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`.
- Passed: `uv run --no-sync python -m compileall -q docs/uad/script/build_hunyuan_real_request_routing_html.py docs/uad/script/build_hunyuan_routing_tabs_html.py docs/uad/script/profile_hunyuan_a13b_routing.py docs/uad/script/profile_hunyuan_official_clean_routing.py docs/uad/script/profile_hunyuan_official_routing.py docs/uad/script/profile_hunyuan_real_request_routing.py vllm_omni/model_executor/models/hunyuan_image3/moe_route_trace.py`.
- Passed: rebuilt a real-request routing HTML report from local traces with `build_hunyuan_real_request_routing_html.py`.
- Passed: rebuilt a combined vLLM/official routing tabs HTML report with `build_hunyuan_routing_tabs_html.py`.
- Passed: `git diff --check -- .gitignore PROGRESS.md docs/uad/script vllm_omni/model_executor/models/hunyuan_image3 vllm_omni/diffusion/models/hunyuan_image3`.

Commit and push:

- Commit `86e29c55` is pushed to the current branch upstream `fork/uad`.

## Experiment Plan: HunyuanImage3 Phase Imbalance

Status: completed.

Completed modifications:

- Created a separate experiment worktree at `/mnt/raid0nvme0/yangshen/code/vllm-omni-uad-phase-util-exp`.
- Fetched latest upstream and merged `origin/main` (`c99df1eb`) into the experiment branch.
- Resolved the HunyuanImage3 pipeline merge conflict by preserving upstream AR KV reuse / custom output changes and
  keeping the opt-in MoE routing trace wrapper.
- Added `docs/uad/script/check_hunyuan_phase_batching.py` as Gate 0 for HunyuanImage3 phase-internal batching.
- Added `docs/uad/phase_imbalance_experiment_plan.md` with dataset, workload, metrics, deployment, and execution plan.

Validation:

- Passed: `python docs/uad/script/check_hunyuan_phase_batching.py --output-json artifacts/uad_phase_imbalance/preflight/phase_batching.json`.
- Passed: `uv run --no-sync ruff check docs/uad/script/check_hunyuan_phase_batching.py`.
- Passed: `uv run --no-sync python -m compileall -q docs/uad/script/check_hunyuan_phase_batching.py`.
- Gate 0 result: `full_phase_internal_continuous_batching_ready=false`.
  AR model code can batch, but the default deploy config keeps `max_num_seqs=1`.
  HunyuanImage3 DiT has no stepwise execution hooks, so diffusion request mode still runs one request at a time.

Commit and push:

- Commit `9c436dfa` is pushed to `fork/uad-phase-util-exp`.
