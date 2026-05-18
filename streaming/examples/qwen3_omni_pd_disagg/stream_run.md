# Qwen3-Omni PD-Disaggregated 流式 Serve & Debug

四阶段 PD 解耦流水线（thinker prefill → thinker decode → talker → code2wav），
默认 `async_chunk` 开启 → 直接支持 SSE 流式 chat。

## 文件

| 文件 | 作用 |
|---|---|
| `qwen3_omni_pd_disagg.yaml` | 4-stage `stage_args` 配置（GPU 0/1/2/2，Mooncake KV 连接） |
| `run_server_disagg.sh` | 单进程或单 stage 启动器，可选 debugpy wait-for-client |
| `run_all_stages_disagg.sh` | 一键拉 4 个 stage，每 stage 独立 debugpy 端口 |
| `run_curl_streaming_disagg.sh` | curl SSE 流式 client |
| `../../../.vscode/launch.json` | VSCode attach 配置（含 compound） |

## 1. 跑通（不调试）

```bash
cd examples/online_serving/qwen3_omni

# 方式 A：单进程 orchestrator（stage 子进程由 mp.Process 起）
./run_server_disagg.sh

# 方式 B：4 个顶级 stage 进程，便于分别 attach
./run_all_stages_disagg.sh
```

## 2. 发流式请求

调用形式：`./run_curl_streaming_disagg.sh [QUERY_TYPE] [MODALITIES]`

- `QUERY_TYPE`：`text` / `use_audio` / `use_image` / `use_video`（默认 `text`），决定输入素材。
- `MODALITIES`：JSON 数组（默认 `null` = 模型默认 text+audio），决定输出走到哪一级 stage。
- env：`HOST` / `PORT` / `MODEL` 改目标，`RAW=1` 关解析直接打印原始 SSE frame。

另一个终端：

```bash
./run_curl_streaming_disagg.sh                                # 最快冒烟测：纯文本 in/out
./run_curl_streaming_disagg.sh use_image                      # 图像 in，验证 vision encoder + 流式
./run_curl_streaming_disagg.sh use_video '["text","audio"]'   # 视频 in + 显式 text+audio out，跑满 4 stage
./run_curl_streaming_disagg.sh text '["text","audio"]'        # 文本 in + audio out，跑满 4 stage 又不被 video 拖慢
RAW=1 ./run_curl_streaming_disagg.sh                          # 不解析，调 SSE 字段时用
HOST=127.0.0.1 PORT=8091 ./run_curl_streaming_disagg.sh       # 跨机 / 换端口
```

正常应看到 `delta.content` 边生成边打印，最后是 `[finish_reason: stop]` + `[usage: ...]`；
带 audio 时音频 base64 在 `delta.audio.data`（`RAW=1` 可看原始字节）。

## 3. 调试

### 3.1 全 4 个 stage 都进断点

```bash
./run_all_stages_disagg.sh --debug
# 4 个 stage 均 wait-for-client
```

VSCode → 调试面板 → **Compound: Attach all PD-Disagg stages (0..3)** → F5。
4 个会话同时上线，端口 5678 / 5679 / 5680 / 5681 对应 stage 0..3。

### 3.2 只调某个 stage

```bash
./run_all_stages_disagg.sh --debug --debug-stages 1      # 只 stage 1 等附加
```

VSCode 单选 **Attach: PD-Disagg stage 1 (decode, 5679)**。

### 3.3 单进程版调试（只能抓 API server）

```bash
./run_server_disagg.sh --debug
```

VSCode 选 **Attach: Qwen3-Omni PD-Disagg (single-process, debugpy 5678)**。

## 4. 注意点

- `--debug-stages` 没列进去的 stage 不会等，握手照常完成；列进去的全部 attach 后整条流水线才推进。
- debugpy 只触达本进程；TP worker / model_runner 是 stage 内部再 spawn 的孙子进程，本方案抓不到。
- Mooncake bootstrap 端口 25201/25202、orchestrator 端口 26000 默认占用，端口冲突时改 yaml 和 `--master-port`。
- stage 1/2/3 后台日志在 `/tmp/vllm_omni_disagg_stage<N>.log`（`--log-dir` 可改）。
- 改 GPU 拓扑：编辑 yaml 里每个 stage 的 `runtime.devices`；PD 两端 `tensor_parallel_size` 必须一致。

## 5. 常用参数速查

`run_server_disagg.sh` / `run_all_stages_disagg.sh`：

| flag | 默认 | 说明 |
|---|---|---|
| `--model` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 模型 id/path |
| `--port` | `8091` | API server 端口（stage 0） |
| `--stage-configs` | `qwen3_omni_pd_disagg.yaml` | PD-disagg yaml |
| `--master-port` | `26000` | Omni orchestrator 端口 |
| `--debug` / `--debug-port` | off / 5678 | debugpy listen + wait-for-client |
| `--debug-stages` | `0,1,2,3` | 仅 `run_all_stages_disagg.sh` |
| `--debug-ports` | `5678,5679,5680,5681` | 仅 `run_all_stages_disagg.sh` |
