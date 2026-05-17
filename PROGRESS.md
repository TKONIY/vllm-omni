# PROGRESS

## Step 0: Toy HunyuanImage3 UAD Entry

Status: completed.

Completed modifications:

- Added the minimal UAD package skeleton under `vllm_omni/uad/`.
- Added `UADEngine` / `AsyncUADEngine` with request add, scheduler step, runner execution, and output application.
- Added `UADRequestState`, `UADToken`, scheduler items, and output dataclasses for the Step 0 ledger.
- Added the early toy `HunyuanImage3UADAdapter`; it was removed in later runner-first cleanup.
- Added toy `HunyuanImage3UADForConditionalGeneration` at `vllm_omni/uad/model/hunyuan_image3.py`.
- Added Step 0 tests in `tests/uad/test_step0.py`.
- Updated `AGENTS.md` with the stop-for-review, validation, push, and progress-recording rule.

Validation:

- Passed: `ruff check vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad/test_step0.py`.
- Passed: `python -m compileall -q vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad/test_step0.py`.
- Passed: `git diff --check -- AGENTS.md PROGRESS.md vllm_omni/uad vllm_omni/uad/model tests/uad/test_step0.py`.
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
- Extended the request state-update output so runner outputs can carry a request state-machine update.
- Updated `UADEngine` to apply phase updates after scheduled AR tokens are marked computed and new engine/materialized tokens are appended.
- Updated `UADShadowScheduler` to skip `dit_step` requests in Step 2, so pending image context tokens are not accidentally scheduled as AR decode tokens.
- Added `HunyuanImage3UADStateConfig` for HunyuanImage3 special-token rules: `<img_ratio_*>` detection, stage-transition helpers, ratio-index extraction, engine-only token filtering, and toy image-context token construction.
- Updated `HunyuanImage3UADAdapter` so a sampled ratio token appends ratio + toy image context tokens, switches the request to `dit_step`, records ratio metadata, and leaves those new tokens uncomputed until a later cache-commit step.
- Updated `docs/uad/plan_uad.md` Step 2 to explicitly map HunyuanImage3 tokenizer/sampler rules, ignored text-mask metadata, AR KV reuse metadata, and later-step obligations.
- Added Step 2 tests in `tests/uad/test_step2_phase_switch.py`.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
- Passed: `git diff --check -- PROGRESS.md docs/uad/plan_uad.md vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
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

## Step 3: Runner-first Toy DiT Step

Status: completed.

Completed modifications:

- Removed the long-term `HunyuanImage3UADAdapter` execution path from the toy UAD implementation.
- Moved HunyuanImage3 special-token/state-machine helpers into the UAD package.
- Updated `UADRunner` so it directly owns the toy HunyuanImage3 model and state config, executes AR toy items, detects `<img_ratio_*>`, and executes fake DiT step items.
- Updated `UADToyScheduler` so `dit_step` requests produce compute-only DiT schedule items instead of being skipped or treated as AR decode.
- Added Step 3 tests covering fake DiT step progress, no KV/token commit during fake denoise, and one tick containing both AR and DiT items.
- Updated Step 0/2 tests to construct `UADRunner` directly without an adapter.
- Updated `docs/uad/plan_uad.md` to mark Step 3 completed and set Step 4 as the next implementation step.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
- Passed: `git diff --check -- PROGRESS.md docs/uad/plan_uad.md vllm_omni/uad tests/uad`.
- Passed: runner-first cleanup grep for removed adapter symbols returned no matches.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py -q`.

Commit and push:

- Step 3 is committed and pushed to the current branch upstream `origin/uad-code`.

## HunyuanImage3 UAD State Helper Comments

Status: completed.

Completed modifications:

- Added method-level comments/docstrings to the HunyuanImage3 UAD state helper.
- Documented how each helper maps to the original HunyuanImage3 tokenizer/sampler rules.
- Documented the current UAD boundary: pure state-machine helper only, no logits masking, no real DiT/VAE execution, and toy image-context placeholders only.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `git diff --check -- vllm_omni/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py -q`.

## Runner / Model State Machine Split

Status: completed.

Completed modifications:

- Added `UADModelStateMachine` as the generic model-specific phase/output-ledger protocol.
- Added `HunyuanImage3UADStateMachine` so HunyuanImage3 owns `<img_ratio_*>`, engine-only token, toy image-context, and toy DiT-step state rules.
- Updated `UADRunner` to delegate sampled AR token semantics and DiT step completion to the state machine instead of inspecting HunyuanImage3 tokens directly.
- Updated Step 2/3 tests and docs to describe the runner/state-machine split.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `git diff --check -- vllm_omni/uad tests/uad docs/uad/design_uad.md docs/uad/plan_uad.md`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py -q`.

## Scheduler / Runner / State Machine Separation

Status: completed.

Completed modifications:

- Changed `UADRunner` to consume a full `UADSchedulerOutput`, group work by phase, and return semantic-free runner output items.
- Moved model-specific state-machine invocation out of `UADRunner` and into `UADEngine` output processing.
- Kept `HunyuanImage3UADStateMachine` responsible only for turning runner raw outputs into request state deltas.
- Added tests that the runner has no state-machine ownership and that engine/output processing delegates sampled-token semantics to the state machine.
- Updated design and plan docs to make the scheduler -> runner -> output processor/state machine flow explicit for continuous batching.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `git diff --check -- vllm_omni/uad tests/uad docs/uad/design_uad.md docs/uad/plan_uad.md`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py -q`.

## Scheduler-Owned State Update

Status: completed.

Completed modifications:

- Moved request-state application out of `UADEngine` and into `UADToyScheduler.update_from_output()`.
- Made `UADToyScheduler` own requests and the model-specific state machine, matching the vLLM-style scheduler/update path.
- Kept `UADRunner.execute_model()` limited to phase-grouped raw output generation.
- Updated tests so scheduler-driven `update_from_output()` is the only state-update path.
- Simplified `docs/uad/design_uad.md` and `docs/uad/plan_uad.md` to the new `schedule -> runner -> scheduler.update_from_output -> output` shape.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `git diff --check -- vllm_omni/uad tests/uad docs/uad/design_uad.md docs/uad/plan_uad.md`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py -q`.

## Step 5 Foundation: Final DiT Persist

Status: completed.

Completed modifications:

- Updated the toy scheduler so non-final `dit_step` items use `persist=False`, while the final `dit_step` uses `persist=True`.
- Made the final toy DiT step persist all pending engine tokens, including the sampled ratio token and toy image context tokens.
- Updated HunyuanImage3 toy state-machine behavior so final DiT returns the request to `ar_decode` and clears `pending_image_context_commit`.
- Added Step 5 tests for final-DiT scheduling, `num_computed_tokens` advancement, and a toy next-turn AR decode after image context persistence.
- Updated `docs/uad/design_uad.md` and `docs/uad/plan_uad.md` to describe the toy final-persist behavior and keep real vLLM KV slot allocation as later work.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `git diff --check -- PROGRESS.md vllm_omni/uad tests/uad docs/uad/design_uad.md docs/uad/plan_uad.md`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step5_persist.py -q`.

## Step 4: HunyuanImage3 UAD Batch Executor Shell

Status: completed.

Completed modifications:

- Added `HunyuanImage3UADModel` as the toy shared-weight model shell and kept
  `HunyuanImage3UADForConditionalGeneration` as a compatibility alias.
- Added `UADBatchItem`, `UADBatchInputs`, `UADBatchItemOutput`, and `UADBatchOutputs`.
- Updated `UADRunner.execute_model()` to pack a full `UADSchedulerOutput` into one mixed
  batch, call `HunyuanImage3UADModel` once, and scatter raw outputs back by scheduler item order.
- Added DiT runner metadata (`dit_step_index`, `total_dit_steps`) to `UADScheduleItem` so the
  runner does not read request state directly.
- Added Step 4 tests for mixed AR + DiT batch packing, attention/FFN token metadata, and output
  scatter order.
- Updated `docs/uad/design_uad.md` and `docs/uad/plan_uad.md` with the implemented batch shell.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad vllm_omni/uad/model/hunyuan_image3.py tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step4_batch_model.py tests/uad/test_step5_persist.py -q`.

## Step 4 Cleanup: Keep UAD Code Under `vllm_omni/uad`

Status: completed.

Completed modifications:

- Moved the toy HunyuanImage3 UAD model shell to `vllm_omni/uad/model/hunyuan_image3.py`.
- Moved generic batch contracts to `vllm_omni/uad/batch.py`.
- Removed the UAD model shell from `vllm_omni/model_executor/models/hunyuan_image3/`.
- Updated runner, tests, and docs to import UAD-local modules only.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step4_batch_model.py tests/uad/test_step5_persist.py -q`.
- Passed: `git diff --check -- PROGRESS.md docs/uad/design_uad.md docs/uad/plan_uad.md vllm_omni/uad tests/uad`.

## State Package Cleanup

Status: completed.

Completed modifications:

- Replaced the previous model-state package with `vllm_omni/uad/state/`.
- Added `vllm_omni/uad/state/base.py` with the abstract `UADModelStateMachine` base class.
- Moved HunyuanImage3 request-state transition rules to `vllm_omni/uad/state/hunyuan_image3.py`.
- Made `HunyuanImage3UADStateMachine` inherit `UADModelStateMachine`.
- Updated scheduler, engine, tests, and docs to import state machines from `vllm_omni.uad.state`.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step4_batch_model.py tests/uad/test_step5_persist.py -q`.

## Batch Contract Field Documentation

Status: completed.

Completed modifications:

- Added attribute documentation for `UADBatchItem`, `UADBatchInputs`,
  `UADBatchItemOutput`, and `UADBatchOutputs`.
- Added property docstrings for derived AR/DiT/FFN/attention token counters.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad/batch.py tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad/batch.py tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step4_batch_model.py tests/uad/test_step5_persist.py -q`.

## Phase-Derived Attention Plan Cleanup

Status: completed.

Completed modifications:

- Removed standalone attention selector fields from `UADBatchItem`.
- Kept batch metadata phase-derived: AR/DiT execution recipes are selected from `phase`.
- Added `ar_item_indices` and `dit_item_indices` helpers for phase grouping.
- Updated Step 4 tests and design docs to state that DiT phase implies prefix paged attention
  plus chunk-local bidirectional attention, merged later with LSE.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step4_batch_model.py tests/uad/test_step5_persist.py -q`.

## Plan Future Steps Detail

Status: completed.

Completed modifications:

- Updated `docs/uad/plan_uad.md` so Steps 5-9 each keep target, scope, and validation sections.
- Marked Step 3 as completed in the current status table.

Validation:

- Passed: `git diff --check -- PROGRESS.md docs/uad/plan_uad.md`.

## UAD ScheduleItem Persist Semantics

Status: completed.

Completed modifications:

- Added `UADScheduleItem.persist` as the scheduler-owned context persistence bit.
- Documented every `UADScheduleItem` field in code and in `docs/uad/design_uad.md`, with AR/DiT examples.
- Updated scheduler state application so `persist=True` advances `num_computed_tokens`; model state machines no longer return a computed-token delta.
- Marked current toy AR schedule items as `persist=True` and toy DiT denoise items as `persist=False`.
- Updated `docs/uad/plan_uad.md` Step 5 to use final `dit_step(persist=True)` instead of adding another phase.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `git diff --check -- vllm_omni/uad tests/uad docs/uad/design_uad.md docs/uad/plan_uad.md`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py -q`.

## UAD Runner / Model Shell Docs

Status: completed.

Completed modifications:

- Rewrote `docs/uad/design_uad.md` so `UADRunner` is the batch-first orchestrator and
  `HunyuanImage3UADModel` is the single shared-weight model shell.
- Updated `docs/uad/plan_uad.md` Step 4 to a unified batch executor shell instead of the old
  per-request DiT framing.
- Kept the state machine as request-state logic only; it does not own batch packing or model forward.

Validation:

- Passed: `git diff --check -- PROGRESS.md docs/uad/design_uad.md docs/uad/plan_uad.md`.

## UAD Output Naming Alignment

Status: completed.

Completed modifications:

- Renamed runner batch output to `UADModelRunnerOutput`, matching vLLM's `ModelRunnerOutput` layer.
- Renamed per-scheduled-item runner output to `UADModelRunnerItemOutput`.
- Renamed model-specific request delta to `UADStateUpdate`.
- Renamed scheduler-updated engine output to `UADEngineCoreOutputs`, matching vLLM's engine-core output layer.
- Updated scheduler, runner, engine, state machine, tests, and UAD docs to remove the older
  pre-vLLM-aligned output names.

Validation:

- Passed: `uv run --no-sync ruff check vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m compileall -q vllm_omni/uad tests/uad`.
- Passed: `uv run --no-sync python -m pytest tests/uad/test_step0.py tests/uad/test_step1_scheduler.py tests/uad/test_step2_phase_switch.py tests/uad/test_step3_runner.py tests/uad/test_step4_batch_model.py tests/uad/test_step5_persist.py -q`.

## UAD Milestone Plan Rewrite

Status: completed.

Completed modifications:

- Rewrote `docs/uad/plan_uad.md` from linear toy steps into milestone-based implementation plan.
- Defined the first complete HunyuanImage3 run boundary: real AR, real AR->DiT transition,
  real DiT denoise, final image context persist, VAE decode, and image artifact output.
- Split remaining work by difficulty into state/metadata, real AR, paged KV persist, real DiT
  attention and denoise, VAE output, continuous batching, mixed FFN/MoE, distributed integration,
  and motivation experiments.
- Added per-milestone implementation steps, validation checks, and review stop points.

Validation:

- Passed: `git diff --check -- PROGRESS.md docs/uad/plan_uad.md`.
