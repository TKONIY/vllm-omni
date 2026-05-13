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
