# DreamZero Quick Start

This document is the shortest path to launching the DreamZero service and connecting the compatible client.

The commands below assume you run them from the repository root.

---

## 1. Start the vLLM DreamZero server

Default example: official HF model + `CF_P=2`.

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

If you only have 1 GPU:

- change `CUDA_VISIBLE_DEVICES=0,1` to `CUDA_VISIBLE_DEVICES=0`
- remove `--cfg-parallel-size 2`

OpenPI WebSocket endpoint:

- `ws://127.0.0.1:8000/v1/realtime/robot/openpi`

---

## 2. Connect the client to the vLLM server

Use the copied DreamZero-compatible client:

- `tests/dreamzero/test_client_AR.py`

When connecting to vLLM, pass the OpenPI websocket path explicitly:

```bash
python tests/dreamzero/test_client_AR.py \
  --host 127.0.0.1 \
  --port 8000 \
  --path /v1/realtime/robot/openpi
```

---

## 3. Connect the same client to the original DreamZero server

The same client defaults to `path=""`, which matches the upstream DreamZero websocket server root path:

```bash
python tests/dreamzero/test_client_AR.py \
  --host 127.0.0.1 \
  --port 8000
```

The only expected protocol difference is the websocket path:

- upstream DreamZero: `ws://HOST:PORT`
- vLLM OpenPI: `ws://HOST:PORT/v1/realtime/robot/openpi`

The client-side observation / infer / reset logic is kept the same for both.

---

## 4. Precision baseline

The currently validated strict-parity baseline is:

- upstream DreamZero in eager mode
- no `torch.compile`
- no DiT cache / skip schedule
- `TP=1`
- `CF_P=1` or `CF_P=2`

Current status:

- `TP=1, CF_P=1`: strict parity
- `TP=1, CF_P=2`: strict parity
- `TP=2, CF_P=1/2`: runs, but strict numerical parity is not guaranteed

---

## 5. Recommended first run

If you want the least surprising setup, start with:

- `GEAR-Dreams/DreamZero-DROID`
- `--enforce-eager`
- `TP=1`
- `CF_P=1`

Then move to `CF_P=2` if you want CFG parallel.

---

## 6. Formal end-to-end parity test

The formal server-vs-server parity test is:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/dreamzero/test_openpi_e2e_source_parity.py -q
```

This test checks:

- upstream DreamZero server
- vLLM DreamZero server
- the same DreamZero-compatible client logic
- strict action-output parity under the non-TP, non-compile baseline

---

## 7. Related docs

- `docs/models/dreamzero/dreamzero.md`: inference call chain
- `docs/models/dreamzero/review.md`: implementation review and conclusions
- `docs/models/dreamzero/todo.md`: remaining issues and precision appendix
