# Stream-Native Engine Plan

Status: discussion draft
Owner: Yangshen
Date: 2026-05-17

This document proposes a plan for building a stream-native runtime for
vLLM-Omni. The goal is not to patch one more special case into `async_chunk`;
it is to define an engine where chunks, deadlines, playback buffers, session
updates, and cross-stage dependencies are first-class scheduling objects.

## Starting Point

The current system already has useful transport and model execution building
blocks, but its control plane is still request-native:

- PR [#727](https://github.com/vllm-project/vllm-omni/pull/727) added
  async chunking across Qwen3-Omni stages and reported a single-request E2E
  reduction from about 3869 ms to about 1305 ms. Its RFC
  [#268](https://github.com/vllm-project/vllm-omni/issues/268) explicitly left
  cross-stage prefill pipeline and async put/get as later work.
- RFC [#3509](https://github.com/vllm-project/vllm-omni/issues/3509) targets
  async D2H/H2D, pinned pools, and a lower-copy SHM wire path. It is transport
  hardening, not a scheduling-policy redesign.
- RFC [#3535](https://github.com/vllm-project/vllm-omni/issues/3535) shows the
  user-visible gap: Qwen3-TTS Base voice_clone at concurrency 64/128/256 has
  underrun p99 around 16.2/17.7/19.1 s against a 100 ms target. That RFC calls
  scheduler/SLA work out of scope.
- PR [#3485](https://github.com/vllm-project/vllm-omni/pull/3485) fixed a TTS
  latency regression and added first-chunk-only `initial_codec_chunk_frames`,
  but the trigger is still a local heuristic.
- PR [#3322](https://github.com/vllm-project/vllm-omni/pull/3322) restores
  Qwen3-TTS Code2Wav cross-request batching. This is important throughput work,
  but it still leaves active-stream selection and underrun deadlines to a future
  scheduler.

Important local code paths:

- `vllm_omni/engine/orchestrator.py` owns request progression. In normal mode it
  forwards only when a stage finishes; in `async_chunk` mode it prewarms
  downstream stages with placeholder prompt lengths.
- `vllm_omni/core/sched/omni_ar_scheduler.py` and
  `vllm_omni/core/sched/omni_generation_scheduler.py` remain vLLM request
  schedulers with Omni hooks around them.
- `vllm_omni/core/sched/omni_scheduling_coordinator.py`,
  `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`,
  and `vllm_omni/worker/omni_connector_model_runner_mixin.py` manage
  `WAITING_FOR_CHUNK`, chunk ready signals, chunk IDs, and connector I/O.
- `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py` implements
  dynamic first chunk using only active request count and stage capacity.
  `qwen3_omni.py` still uses fixed `codec_chunk_frames` for Talker to
  Code2Wav.
- `vllm_omni/entrypoints/openai/api_server.py` rejects `/v1/realtime` when
  `async_chunk` is enabled.

## Why A New Engine

The missing abstraction is a stream graph, not a faster connector. A stream
chunk has an upstream producer, downstream credit, an accumulated payload size,
a session/request identity, a deadline derived from SLO and playback state, and
a completion/merge policy. Today those are split across model-specific
processors, scheduler status mutations, queue polling, and frontend code.

That split creates four structural limits:

- Scheduling is stage-local and SLO-unaware. A downstream Code2Wav stage sees
  ready requests, not chunks with deadlines or playback debt.
- Emission policy is embedded in processors. A config like
  `initial_codec_chunk_frames=1` can improve TTFA but has no visibility into
  downstream queue depth, user underrun, or overload.
- Prefill remains a barrier across stages. Intra-stage chunked prefill exists,
  but encoder/Thinker/Talker dependencies are not represented as incremental
  downstream work.
- Bounded requests and unbounded sessions are different serving modes. The API
  layer currently prevents `async_chunk` and `/v1/realtime` from coexisting.

## Target Runtime

The stream-native engine should make the unit of scheduling a typed stream
event:

```text
StreamEvent {
  stream_id, session_id, request_id,
  stage_id, edge_id, seq_no,
  kind: prefill | decode | codec | waveform | control,
  payload_ref,
  ready_time, deadline, slo_class,
  upstream_progress, downstream_credit,
  size_hint, cost_hint,
  policy_state
}
```

The runtime then has five core components:

- **Stream graph registry**: a DAG of stage actors and edges. It replaces
  hardcoded `stage_id + 1` progression with explicit producer/consumer
  dependencies.
- **Trigger policy engine**: per-edge `emit | hold | merge` decisions based on
  deadline, downstream credit, upstream progress, accumulated size, and quality
  constraints.
- **Global chunk scheduler**: EDF or weighted EDF over ready stream events, with
  per-stage resource constraints, SLO classes, admission control, and overload
  degradation.
- **Credit/backpressure manager**: per-edge and per-stream credits. Credits are
  owned by the consumer and consumed by the producer before emitting.
- **Unified session/request lifecycle**: bounded TTS/chat requests and unbounded
  realtime/VLA sessions are both streams with different end conditions.

## What To Reuse

Reuse these pieces aggressively:

- Model weights, model classes, tokenizers, sampling params, config parsing, and
  stage YAML schema where possible.
- vLLM attention kernels, KV cache/block managers, CUDA graph capture, LoRA,
  prefix-cache primitives after their correctness issues are resolved.
- Existing stage model runners as a compatibility layer for executing a batch
  once the new scheduler has chosen work. Over time the runner input should move
  from request-shaped `SchedulerOutput` to stream-event batches.
- Omni connectors as the transport substrate, especially after #3509-style
  async D2H/H2D and tensor-blob hardening. The connector should remain a data
  plane, not a scheduler.
- Model-specific packing/unpacking logic in stage processors. Keep tensor
  transforms, speaker/language metadata extraction, and Code2Wav context-window
  construction; move trigger decisions out.
- Serving protocol objects and output processors where they are pure protocol
  translation.
- Existing benchmark and stability harnesses under `vllm_omni/benchmarks`,
  `tests/dfx/perf`, and `tests/dfx/stability`.

## What To Rewrite Or Drop

Rewrite these surfaces instead of stretching them further:

- **Orchestrator request progression**:
  `Orchestrator._route_output`, `_forward_to_next_stage`, and
  `_prewarm_async_chunk_stages` should be replaced by stream graph activation.
  Placeholder downstream requests and `compute_talker_prompt_ids_length()`
  prewarm are compatibility hacks.
- **Stage-local request schedulers as the policy authority**:
  keep vLLM scheduling machinery for GPU execution, but do not make
  `OmniARScheduler` / `OmniGenerationScheduler` responsible for chunk readiness,
  SLO policy, or inter-stage fairness.
- **`WAITING_FOR_CHUNK` as the main abstraction**:
  request status mutation is too coarse. A request/session may have multiple
  ready and blocked chunks on different edges.
- **`OmniChunkTransferAdapter` as lifecycle owner**:
  its chunk ID, finished set, request payload, and queue state should move into
  a stream runtime. The adapter/mixin should become transport-only plus metrics.
- **Model processors as trigger policies**:
  functions like `talker2code2wav_async_chunk` should be split into
  `pack_window(...)` and policy-free tensor transforms. A separate trigger policy
  chooses when and how large a window should be.
- **Full-payload special cases**:
  `WAITING_FOR_INPUT` and Qwen3 full-payload coordinator logic should become a
  stream edge mode: terminal payload is just one event kind.
- **API mutual exclusion**:
  remove the server-level `async_chunk` versus `/v1/realtime` split. Realtime is
  a long-lived stream, not a separate engine mode.
- **Unbounded SHM queues**:
  producer emission without consumer credit should be disallowed in the new
  runtime.

## Step Plan

### Phase 0: Measurement Contract

Before implementing scheduling, pin the workload and metrics.

- Add continuity metrics to the benchmark output: underrun p50/p95/p99,
  playback buffer depth, chunk arrival gap, stage queue depth, emitted chunk
  size, and per-stage service time.
- Reproduce current baselines for Qwen3-TTS Base voice_clone and CustomVoice at
  c=32/64/128/256, plus Qwen3-Omni audio at c=1/4/10/32.
- Add mixed traffic: text-only, text+audio, voice_clone, and realtime audio
  sessions in one run.
- Record traces that separate transport wait, scheduler wait, stage compute,
  D2H/H2D, and client playback underrun.

Exit criterion: one command produces the current "why this is broken" table and
can compare any later policy.

### Phase 1: Stream Runtime Skeleton

Build a side-by-side prototype without deleting the existing engine.

- Introduce `vllm_omni/streaming/` with `StreamEvent`, `StreamGraph`,
  `StreamState`, `CreditState`, and `TriggerPolicy`.
- Implement an in-process simulator backend first, using measured per-stage
  service distributions from Phase 0.
- Add adapters from existing scheduler/model-runner outputs into stream events
  for Qwen3-TTS only.
- Keep existing transport and execution path; the stream runtime only decides
  when to emit Talker to Code2Wav chunks.

Exit criterion: Qwen3-TTS can run with old execution but new trigger decisions
behind a feature flag.

### Phase 2: Credit-Based Transport Boundary

Make the producer/consumer contract explicit.

- Add per-edge credits: `available_chunks`, `available_bytes`, and optional
  `target_buffer_ms`.
- Stage 1/Code2Wav returns credits after a chunk is scheduled or decoded.
- Stage 0/Talker cannot enqueue new chunks without credit, except terminal
  control events.
- Export queue depth and credit metrics.
- Keep the old connector implementation, but remove unbounded producer growth
  from the tested path.

Exit criterion: SHM depth is bounded under c=128/256 and no request can build an
unlimited backlog.

### Phase 3: Deadline-Aware Chunk Scheduling

Move from readiness to policy.

- Add per-stream SLO state: TTFA deadline, steady-state underrun deadline,
  max chunk gap, and SLO class.
- Implement EDF over chunks within each stage, then weighted EDF across SLO
  classes.
- Add policy knobs:
  - low-latency: small first chunk, aggressive Code2Wav scheduling;
  - balanced: meet playback underrun while preserving throughput;
  - throughput: larger chunks and fewer context-window decodes.
- Add overload behavior: admission control, downgrade noninteractive streams,
  and merge chunks when deadlines are already missed.

Exit criterion: the same workload has higher SLO goodput than round-robin at the
same GPU budget, and overload degrades predictably.

### Phase 4: Unified Request And Session Runtime

Remove the `async_chunk` / realtime split.

- Represent `/v1/audio/speech`, `/v1/chat/completions`, `/v1/realtime`, video
  stream, and VLA sessions as stream sessions.
- Replace server-level rejection of realtime under `async_chunk` with session
  admission into the stream runtime.
- Support session updates as control events that append to or revise stream
  state.
- Define cancellation, draining, and terminal sentinels at stream-event level.

Exit criterion: bounded TTS requests and one realtime session can share the same
scheduler and stages in one server process.

### Phase 5: Cross-Stage Prefill Streaming

Address the input-side barrier.

- Split encoder/Thinker prefill outputs into incremental semantic chunks.
- Let Talker prefill begin on partial Thinker state with explicit dependency
  ranges.
- Add policy to decide whether a partial prefill chunk is worth emitting based
  on remaining prompt length, expected TTFA benefit, and downstream load.
- Validate quality and alignment for audio/video inputs, especially where
  future context changes model behavior.

Exit criterion: long audio/video prompt TTFA improves without measurable quality
regression on selected accuracy tests.

### Phase 6: Production Hardening

- Multi-replica sticky routing and credit sharing.
- Fault isolation per replica instead of fail-stop for all stages.
- 4h stability runs with bounded `/dev/shm`, host RSS, pinned memory, and GPU
  memory.
- Compatibility bridge for legacy stage configs.
- Feature flags and rollback at phase boundaries.

## Motivation Experiments

These experiments should be run before and during implementation, not only at
the end.

1. **Continuity baseline**: Qwen3-TTS Base voice_clone and CustomVoice at
   c=32/64/128/256. Metrics: underrun p95/p99, TTFA, RTF, request goodput,
   stage utilization, SHM depth.
2. **Trigger ablation**: fixed `initial_codec_chunk_frames`
   `{1,2,4,8,16,25}`, current dynamic IC, deadline-aware IC, and
   credit-aware merge. Metrics: TTFA, underrun, Code2Wav compute amplification,
   audio quality.
3. **Scheduler ablation**: current round-robin/request order, FIFO by chunk
   arrival, EDF, weighted EDF, EDF plus credit, EDF plus admission control.
   Use mixed SLO classes.
4. **Credit ablation**: unbounded queue, static credit, adaptive credit by
   playback buffer, adaptive credit by consumer service time. Metrics: queue
   depth, memory, underrun, producer idle time.
5. **Transport versus policy isolation**: baseline, #3509-style async D2H/H2D
   only, policy only, and combined. This proves whether the remaining gap is
   scheduling rather than copy overhead.
6. **Cross-stage prefill barrier**: full Thinker prefill before Talker, fixed
   chunked prefill, and deadline-triggered prefill. Use long audio/video input
   and measure TTFA, ITL, quality/WER, and stage overlap.
7. **Unified sessions**: mix short TTS requests, long chat+audio requests, and
   one or more `/v1/realtime` sessions. Verify no API-mode exclusion and measure
   fairness.
8. **Multi-replica routing**: one Code2Wav replica versus two or more with sticky
   per-stream routing. Measure RTF, underrun, and routing correctness.
9. **Overload behavior**: drive above capacity and verify admission/downgrade
   decisions improve SLO goodput rather than letting every stream miss.

Primary result metric should be **SLO goodput**: requests or active sessions per
second satisfying all relevant constraints, including TTFA, steady-state
underrun, text ITL, RTF, and success rate.

## Open Questions

- What chunk deadline model should drive EDF: absolute audio playback deadline,
  relative inter-chunk gap, or a hybrid with TTFA and steady-state phases?
- How much partial prefill is safe for Qwen3-Omni quality, especially for video
  and audio where future context may change semantic state?
- Should stream scheduling live in a new engine process, or inside the existing
  orchestrator thread during the transition?
- How do we expose operator policy without making every model YAML carry a
  scheduler research project?
- Which legacy APIs must be kept byte-compatible during the first prototype?

## Suggested First PRs

1. Add benchmark continuity metrics and workload configs.
2. Add `vllm_omni/streaming/` data types plus a simulator test.
3. Split Qwen3-TTS `talker2code2wav_async_chunk` into trigger-free packing plus
   the current trigger policy wrapper.
4. Add a feature-flagged credit gate on Talker to Code2Wav emission.
5. Add EDF policy in the simulator, then attach it to Qwen3-TTS behind a flag.
