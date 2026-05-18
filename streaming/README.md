# streaming/

Working bundle for the stream-native engine effort on vLLM-Omni — design notes,
call-stack documentation, and runnable PD-disaggregated streaming serve
examples for Qwen3-Omni.

```
streaming/
├── design/                              # design docs & architecture notes
│   ├── stream_native_engine_plan.md
│   ├── stream_native_engine_plan_zh.md
│   └── qwen_call_stack.html             # QwenOmni / QwenTTS call-stack reference
└── examples/
    └── qwen3_omni_pd_disagg/            # Qwen3-Omni streaming serve recipes
        ├── qwen3_omni_pd_disagg.yaml    # 4-stage PD-disagg (H100/A100-80GB)
        ├── qwen3_omni_a5000.yaml        # 3-stage non-disagg (8x A5000-24GB)
        ├── run_server_disagg.sh         # single-process / per-stage launcher
        ├── run_all_stages_disagg.sh     # 4-stage PD multi-process launcher
        ├── run_all_stages_nondisagg.sh  # 3-stage non-disagg multi-process launcher
        ├── run_curl_streaming_disagg.sh # SSE streaming client
        └── stream_run.md
```

## Design

| File | What |
|---|---|
| [`stream_native_engine_plan.md`](design/stream_native_engine_plan.md)       | Stream-native engine plan (English, discussion draft) |
| [`stream_native_engine_plan_zh.md`](design/stream_native_engine_plan_zh.md) | Stream-native engine plan (中文版) |
| [`qwen_call_stack.html`](design/qwen_call_stack.html)                       | Interactive HTML walkthrough of QwenOmni & QwenTTS pipelines — entry to streaming output, including sync vs async-chunk data paths. Open in a browser. |

## Examples

`examples/qwen3_omni_pd_disagg/` ships two streaming-serve flavours of
Qwen3-Omni; both keep `async_chunk` on so `stream=true` chat requests
work end-to-end:

| Variant | Stages | YAML | Multi-proc launcher | Target hardware |
|---|---|---|---|---|
| PD-disagg | thinker prefill → thinker decode → talker → code2wav (4) | `qwen3_omni_pd_disagg.yaml` | `run_all_stages_disagg.sh` | H100 / A100-80GB |
| Non-disagg | thinker (merged) → talker → code2wav (3) | `qwen3_omni_a5000.yaml` | `run_all_stages_nondisagg.sh` | 8x A5000-24GB (TP=4 on thinker) |

The PD variant uses Mooncake KV transport between the two thinker
stages; the non-disagg variant drops Mooncake and merges thinker
prefill+decode so the 30B MoE fits at TP=4 on 24 GB cards.

```bash
cd streaming/examples/qwen3_omni_pd_disagg

# PD-disagg (4 stages):
./run_all_stages_disagg.sh

# Non-disagg (3 stages, A5000-friendly):
./run_all_stages_nondisagg.sh

# SSE streaming client (shared):
./run_curl_streaming_disagg.sh
```

See [`stream_run.md`](examples/qwen3_omni_pd_disagg/stream_run.md) for the
full debug/serve recipe, including VSCode attach configuration.
