# DreamZero Usage

This document is for users who want to run the DreamZero service in `vllm-omni-wm`. It focuses on:

- what is currently supported
- how to launch it
- how `--model` / `model_path` can be specified
- where each component is loaded from under different input forms
- what kind of weight directory is expected
- current precision / behavior boundaries

---

## 0. Quick Start: Server + Client

### 0.1 Start the vLLM DreamZero server

The default example below uses the official HF model name and starts with `CF_P=2`:

```bash
ATTENTION_BACKEND=torch \
DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29628 \
vllm serve \
  GEAR-Dreams/DreamZero-DROID \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --cfg-parallel-size 2 \
  --enforce-eager
```

If you only have 1 GPU, change `CUDA_VISIBLE_DEVICES=0,1` to `CUDA_VISIBLE_DEVICES=0` and remove `--cfg-parallel-size 2`.

OpenPI WebSocket endpoint:

- `ws://127.0.0.1:8000/v1/realtime/robot/openpi`

### 0.2 Start the client and connect to the vLLM server

The repo keeps a client copied from DreamZero with path adaptation:

- `tests/dreamzero/test_client_AR.py`

By default it connects to the original DreamZero server root path. When connecting to vLLM, pass:

- `--path /v1/realtime/robot/openpi`

Run:

```bash
python tests/dreamzero/test_client_AR.py \
  --host 127.0.0.1 \
  --port 8000 \
  --path /v1/realtime/robot/openpi
```

### 0.3 If you want to connect to the original DreamZero server

`tests/dreamzero/test_client_AR.py` defaults to `path=""`, so for the upstream server:

```bash
python tests/dreamzero/test_client_AR.py \
  --host 127.0.0.1 \
  --port 8000
```

---

## 1. Current Support Scope

### 1.1 Supported

- a single `DreamZeroPipeline`
- official OpenPI WebSocket serving:
  - `/v1/realtime/robot/openpi`
- direct startup from official DreamZero HF roots:
  - `GEAR-Dreams/DreamZero-DROID`
  - `GEAR-Dreams/DreamZero-AgiBot`
- startup from a local DreamZero root directory
- startup from a local prepared bundle directory
- `TP=1`
- `CF_P=1/2`
- `TP=2` runs

### 1.2 Current precision contract

- **Strict parity contract**: DreamZero upstream in **eager** mode, **without `torch.compile`**, and **without DiT cache / skip schedule**
- Confirmed:
  - `TP=1, CF_P=1`: strict parity
  - `TP=1, CF_P=2`: strict parity
  - `TP=2, CF_P=1/2`: **runs, but strict numerical parity is not guaranteed**

### 1.3 Not implemented / not reconnected yet

- DiT cache / skip schedule
- 2-stage pipeline
- sequence parallel / Ulysses / ring parallel
- parity with DreamZero upstream compiled scheduler behavior

---

## 2. Minimal Startup Modes

### 2.1 Official HF repo name

Recommended command (2 GPUs / `CF_P=2`):

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/vllm serve \
  GEAR-Dreams/DreamZero-DROID \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --cfg-parallel-size 2 \
  --enforce-eager
```

Single-GPU version:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve \
  GEAR-Dreams/DreamZero-DROID \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --enforce-eager
```

AgiBot:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve \
  GEAR-Dreams/DreamZero-AgiBot \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-agibot \
  --enforce-eager
```

Serving entrypoint:

- WebSocket: `ws://HOST:PORT/v1/realtime/robot/openpi`

### 2.2 Local root directory

If you already have a local DreamZero root, you can run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve \
  <dreamzero-root> \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --enforce-eager
```

### 2.3 Recommended: always use `--enforce-eager`

All current “strict parity with the original DreamZero repo” statements are based on:

- `--enforce-eager`
- no DiT cache / skip schedule

Do not interpret the current implementation as “already matched to the DreamZero compiled path”.

---

## 3. The Three Ways to Specify `--model` / `model_path`

### 3.1 Option A: official HF repo id

Examples:

- `GEAR-Dreams/DreamZero-DROID`
- `GEAR-Dreams/DreamZero-AgiBot`

This is the **recommended mode**.

#### What you need

- no manual local directory preparation
- internet access, or a pre-warmed HF cache

#### Current behavior

- `vllm serve` automatically downloads the repo
- for a cold start, you should currently treat this as downloading the **entire official repo snapshot**
- local observation shows the full snapshot cache is about `61G`

#### Best for

- the standard user flow
- the closest match to the target contract of “just provide a model name and start serving”

### 3.2 Option B: local HF snapshot / local root mirror

Examples:

- `<hf-cache>/models--GEAR-Dreams--DreamZero-DROID/snapshots/<sha>`
- an equivalent root directory you downloaded or synced yourself

#### Minimum required structure

```text
config.json
experiment_cfg/metadata.json
model.safetensors.index.json
model-00001-of-00010.safetensors
...
model-00010-of-00010.safetensors
```

#### Best for

- offline environments
- pinning to a specific snapshot
- avoiding the initial cold download

### 3.3 Option C: local prepared bundle

Example:

- `<prepared-dreamzero-bundle>`

These directories usually contain the DreamZero root files plus extra symlinks or helper subdirectories such as:

- `vae/`
- `tokenizer/`
- `text_encoder/`

#### Current behavior

- **supported**
- but these extra subdirectories are now **not a hard requirement** for DreamZero startup
- the current implementation only treats such layouts as a compatible input form

#### Best for

- backward compatibility
- local debugging
- reusing an existing DreamZero working directory

---

## 4. Where Each Component Is Loaded From

### 4.1 General rule

The current goal for DreamZero in vLLM is:

- **the final learned weights must come from the DreamZero root checkpoint**

That means these four groups should ultimately come from the root `model-*.safetensors`:

- `action_head.model.*`
- `action_head.text_encoder.*`
- `action_head.image_encoder.*`
- `action_head.vae.*`

### 4.2 Component loading table

| Component | Initialization source | Final weight source | Notes |
| --- | --- | --- | --- |
| tokenizer | `google/umt5-xxl` by default, or `model_paths["tokenizer"]` | tokenizer repo / local tokenizer path | DreamZero root does not include a tokenizer subdirectory |
| text_encoder | local `UMT5EncoderModel(config)` | DreamZero root `action_head.text_encoder.*` | not loaded from a root `text_encoder/` subdirectory |
| image_encoder | local `DreamZeroImageEncoder()` | DreamZero root `action_head.image_encoder.*` | uses the source-shaped implementation, not HF `CLIPVisionModel` in the production path |
| vae | `model_paths["vae"]` if provided; else `model/vae` if present locally; else direct `DistributedAutoencoderKLWan()` | DreamZero root `action_head.vae.*` | `model_paths["vae"]` only changes the skeleton source, not the final learned-weight truth |
| transformer | local `CausalWanModel(config)` | DreamZero root `action_head.model.*` | backbone config comes from root `config.json` |
| metadata | root `experiment_cfg/metadata.json` | root `experiment_cfg/metadata.json` | used for action/state normalization |

### 4.3 One-line summary for `vae`

You no longer need an extra `vae/` subdirectory when launching from the official HF root.

However:

- if you explicitly provide `model_paths["vae"]`
- or if your local directory already contains `vae/`

the current implementation will still accept it.

---

## 5. Root `config.json` Semantics That Must Be Preserved

DreamZero auto-detection depends on DreamZero markers inside the root `config.json`.

The current auto-detection logic expects the local/remote root to still keep information like:

- `model_type == "vla"`
- `_target_` values under `action_head_cfg` / `backbone_cfg` still containing the `dreamzero` namespace

If you heavily edited `config.json` and removed those DreamZero markers, auto-detection may fail.

---

## 6. Configurable Options

It is helpful to separate them into two groups:

- framework-level startup arguments
- DreamZero-specific / programmatic overrides

### 6.1 Framework-level startup arguments

The most commonly used ones are:

- `--model`
  - DreamZero root checkpoint
- `--served-model-name`
  - recommended:
    - `dreamzero-droid` for DROID
    - `dreamzero-agibot` for AgiBot
- `--omni`
  - required
- `--enforce-eager`
  - **strongly recommended**
- `--tensor-parallel-size`
  - supported, but `TP>1` does not currently guarantee strict precision parity
- `--cfg-parallel-size`
  - supported; pipeline-level validation shows `TP=1, CF_P=2` can match strictly

### 6.2 `model_paths` (advanced)

DreamZero currently reads only two keys:

- `model_paths["tokenizer"]`
  - default: `google/umt5-xxl`
  - useful for offline environments or a local tokenizer mirror
- `model_paths["vae"]`
  - optional
  - only affects the VAE skeleton initialization source
  - **does not change** the fact that the final weights are still overwritten from DreamZero root `action_head.vae.*`

Recommended guidance:

- normal users: **do not set it**
- offline environments: only set a local tokenizer path if needed

### 6.3 `model_config` (advanced)

`DreamZeroPipeline` currently reads these runtime overrides:

- `num_inference_steps`
- `cfg_scale`
- `sigma_shift`
- `seed`
- `embodiment_name_to_id`
- `action_norm_stats_path`
- `relative_action`
- `relative_action_dim`

These are read starting from `vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py:219`.

For the following fields, if they already exist in the DreamZero HF root `config.json`
under `action_head_cfg.config.*`, the current implementation reads them only from the
root config and does not read them from `model_config`, nor fall back to defaults:

- `action_dim`
- `hidden_size`
- `num_frames`
- `num_frame_per_block`
- `action_horizon`
- `decouple_inference_noise`
- `video_inference_final_noise`
- `max_state_dim`
- `max_action_dim`

#### Recommended usage

- **do not change them for normal serving**
- only change them if:
  - root `metadata.json` is missing and you need to provide `action_norm_stats_path`
  - you are doing an offline experiment and explicitly understand that you are changing DreamZero behavior

#### Important note

- the DreamZero eager baseline uses `num_inference_steps = 16`
- the DreamZero root `config.json` **does indeed contain**
  `action_head_cfg.config.num_inference_timesteps = 4`
- this field comes from upstream
  `wan_flow_matching_action_tf.yaml` / `WANPolicyHeadConfig`
- but in the actual upstream inference path, scheduler stepping uses
  `WANPolicyHead.num_inference_steps = 16`, not this `4`
- for parity with upstream eager, the current `vllm-omni` DreamZero path also does **not**
  use `num_inference_timesteps` to drive the service denoise loop
- if you want to run a non-parity experiment, the field you should change is
  `model_config.num_inference_steps`; for normal serving, do not mistake
  `num_inference_timesteps=4` for the serving inference step count
- similarly, for fields that are already explicitly given by the DreamZero root
  `config.json`, such as `num_frames=33`, `action_horizon=24`, and `max_state_dim=64`,
  do not try to override them again through `model_config`

---

## 7. Precision / Behavior Notes

### 7.1 Current strict-parity matrix

Using DreamZero upstream eager, no-compile, and no-DiT-cache/skip as the reference:

| Configuration | Result |
| --- | --- |
| `TP=1, CF_P=1` | strict parity |
| `TP=1, CF_P=2` | strict parity |
| `TP=2, CF_P=1` | runs, but not strictly aligned |
| `TP=2, CF_P=2` | runs, but not strictly aligned |

### 7.2 Why `TP=2` is not fully aligned

The current implementation intentionally keeps the native vLLM path:

- `RowParallelLinear`
- `bf16 x bf16`
- TP all-reduce

This path itself introduces numerical drift under `bf16 + TP>1`.

Current observations:

- in a single-layer focused test:
  - `TP=2 + bf16` shows visible drift in `RowParallelLinear`
- in DreamZero pipeline parity:
  - the first stable failure point is `state.positive.kv[1]`
  - the scale is about `1.562e-02`

So the current documentation contract should be:

- `TP=2`: **functionally supported**
- `TP=2`: **no promise of bitwise / strict allclose parity with upstream eager**

### 7.3 Current CFG parallel contract

CFG parallel is currently wired with the DreamZero formula:

- both prefill and denoise go through the same `predict_noise_maybe_with_cfg()` path
- `cfg_normalize=False`

Current validation result:

- `TP=1, CF_P=2`: strict parity

### 7.4 `torch.compile`

Do not interpret the current DreamZero port as “already matched to the compiled version”.

The only current claim is:

- **matched to DreamZero eager**

DreamZero upstream itself already shows `bf16` numerical differences between:

- eager
- compiled

---

## 8. Not Implemented Yet

The following items are **not** currently reconnected. Do not assume they are available by default:

- DiT cache / static skip mask
- dynamic cache schedule / cosine skip
- 2-stage pipeline
- sequence parallel / ring parallel

If these are reintroduced later, both this usage doc and the precision contract should be updated separately.

---

## 9. Recommended Usage Patterns

### 9.1 If you want the most stable setup

Recommended:

- `--model GEAR-Dreams/DreamZero-DROID`
- `--served-model-name dreamzero-droid`
- `--enforce-eager`
- `TP=1`
- `CF_P=1`

### 9.2 If you want strict parity with upstream eager

Recommended:

- `--enforce-eager`
- do not enable DiT cache / skip schedule
- `TP=1`
- `CF_P=1` or `CF_P=2`

### 9.3 If you want higher throughput

You can try:

- `TP=2`

but you must accept:

- current precision is not fully aligned with upstream eager

---

## 10. Cold-Start Recommendation

If you plan to use the official HF repo id directly in production, it is recommended to pre-download it:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("GEAR-Dreams/DreamZero-DROID")
PY
```

Reason:

- cold start is currently relatively slow
- formal online parity has already verified that after the cache is warm, the official HF repo id can be used directly and still match upstream eager strictly

---

## 11. Cross References

- `docs/models/dreamzero/dreamzero.md`: inference call chain
- `docs/models/dreamzero/review.md`: implementation review and conclusions
- `docs/models/dreamzero/todo.md`: remaining issues and precision appendix
