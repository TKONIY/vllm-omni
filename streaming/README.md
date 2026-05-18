# streaming/

Working bundle for the stream-native engine effort on vLLM-Omni — design notes,
call-stack documentation, and runnable PD-disaggregated streaming serve
examples for Qwen3-Omni.

```
streaming/
├── design/                         # design docs & architecture notes
│   ├── stream_native_engine_plan.md
│   ├── stream_native_engine_plan_zh.md
│   └── qwen_call_stack.html        # QwenOmni / QwenTTS call-stack reference
└── examples/
    └── qwen3_omni_pd_disagg/       # 4-stage PD-disaggregated streaming serve
        ├── qwen3_omni_pd_disagg.yaml
        ├── run_server_disagg.sh
        ├── run_all_stages_disagg.sh
        ├── run_curl_streaming_disagg.sh
        └── stream_run.md
```

## Design

| File | What |
|---|---|
| [`stream_native_engine_plan.md`](design/stream_native_engine_plan.md)       | Stream-native engine plan (English, discussion draft) |
| [`stream_native_engine_plan_zh.md`](design/stream_native_engine_plan_zh.md) | Stream-native engine plan (中文版) |
| [`qwen_call_stack.html`](design/qwen_call_stack.html)                       | Interactive HTML walkthrough of QwenOmni & QwenTTS pipelines — entry to streaming output, including sync vs async-chunk data paths. Open in a browser. |

## Examples

`examples/qwen3_omni_pd_disagg/` runs Qwen3-Omni as four PD-disaggregated
stages (thinker prefill → thinker decode → talker → code2wav) over Mooncake
KV transport, with `async_chunk` on so `stream=true` chat requests work
end-to-end.

```bash
cd streaming/examples/qwen3_omni_pd_disagg
./run_all_stages_disagg.sh         # spawn all four stages
./run_curl_streaming_disagg.sh     # SSE streaming client
```

See [`stream_run.md`](examples/qwen3_omni_pd_disagg/stream_run.md) for the
full debug/serve recipe, including VSCode attach configuration.
