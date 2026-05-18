# Qwen3-Omni 流式 Serve & Debug

两套流水线，都开 `async_chunk` 直接支持 SSE 流式 chat：

- **PD-disagg（4 stage）**：thinker prefill → thinker decode → talker → code2wav，
  两个 thinker stage 间走 Mooncake KV 传输。需要 H100 / A100-80GB 量级单卡显存。
- **Non-disagg（3 stage）**：thinker（prefill+decode 合并）→ talker → code2wav，
  无 Mooncake。为 A5000-24GB 这类小显存卡设计，thinker 用 TP=4 摊到 4 卡。

## 文件

| 文件 | 作用 |
|---|---|
| `qwen3_omni_pd_disagg.yaml` | PD-disagg 4-stage 配置（GPU 0/1/2/2，Mooncake KV 连接） |
| `qwen3_omni_a5000.yaml` | Non-disagg 3-stage 配置（GPU 0-3 TP=4 thinker / GPU 4 talker+code2wav） |
| `run_server_disagg.sh` | 单进程或单 stage 启动器，可选 debugpy wait-for-client。默认 yaml 指向 `qwen3_omni_a5000.yaml`，PD 用 `--stage-configs qwen3_omni_pd_disagg.yaml` 切换 |
| `run_all_stages_disagg.sh` | PD-disagg：一键拉 4 个 stage，每 stage 独立 debugpy 端口（5678..5681） |
| `run_all_stages_nondisagg.sh` | Non-disagg：一键拉 3 个 stage，每 stage 独立 debugpy 端口（5678..5680） |
| `run_curl_streaming_disagg.sh` | curl SSE 流式 client（两个变体共用） |
| `../../../.vscode/launch.json` | VSCode attach 配置（含两个 compound：4-stage / 3-stage） |

## 1. 跑通（不调试）

```bash
cd streaming/examples/qwen3_omni_pd_disagg

# PD-disagg
./run_server_disagg.sh --stage-configs qwen3_omni_pd_disagg.yaml   # 单进程
./run_all_stages_disagg.sh                                         # 4 个顶级 stage

# Non-disagg（A5000 默认）
./run_server_disagg.sh                                             # 单进程
./run_all_stages_nondisagg.sh                                      # 3 个顶级 stage
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

### 3.1 PD-disagg：全 4 个 stage 都进断点

```bash
./run_all_stages_disagg.sh --debug
# 4 个 stage 均 wait-for-client
```

VSCode → 调试面板 → **Compound: Attach all PD-Disagg stages (0..3)** → F5。
4 个会话同时上线，端口 5678 / 5679 / 5680 / 5681 对应 stage 0..3。

### 3.2 Non-disagg：全 3 个 stage 都进断点

```bash
./run_all_stages_nondisagg.sh --debug
# 3 个 stage 均 wait-for-client
```

VSCode → 调试面板 → **Compound: Attach all non-disagg stages (0..2)** → F5。
端口 5678 / 5679 / 5680 对应 stage 0 (thinker) / 1 (talker) / 2 (code2wav)。

### 3.3 只调某个 stage

```bash
# PD-disagg
./run_all_stages_disagg.sh --debug --debug-stages 1      # 只 thinker decode 等附加

# Non-disagg
./run_all_stages_nondisagg.sh --debug --debug-stages 0   # 只 thinker 等附加
```

VSCode 单选对应的 attach 配置（按端口找）。

### 3.4 单进程版调试（只能抓 API server）

```bash
./run_server_disagg.sh --debug                                      # non-disagg (默认 yaml)
./run_server_disagg.sh --stage-configs qwen3_omni_pd_disagg.yaml --debug   # PD-disagg
```

VSCode 选 **Attach: Qwen3-Omni PD-Disagg (single-process, debugpy 5678)**
（名字里 PD-Disagg 是历史叫法，两个变体单进程都用这个 attach）。

## 4. 注意点

- `--debug-stages` 没列进去的 stage 不会等，握手照常完成；列进去的全部 attach 后整条流水线才推进。
- debugpy 只触达本进程；TP worker / model_runner 是 stage 内部再 spawn 的孙子进程，本方案抓不到。
- PD-disagg 用 Mooncake bootstrap 端口 25201/25202；non-disagg 没有这两个端口。orchestrator 端口 26000 两个变体都用，冲突时改 yaml 和 `--master-port`。
- 后台 stage 日志：PD-disagg 在 `/tmp/vllm_omni_disagg_stage<N>.log`，non-disagg 在 `/tmp/vllm_omni_nondisagg_stage<N>.log`（`--log-dir` 可改）。
- 改 GPU 拓扑：编辑 yaml 里每个 stage 的 `runtime.devices`。PD-disagg 两端 `tensor_parallel_size` 必须一致；non-disagg 没这个约束。
- A5000-24GB 上跑 BF16 30B MoE 必须 `tensor_parallel_size: 4`（单卡装不下 ~60 GB 权重）。PD-disagg 需要 thinker 两端各占 4 卡共 8 卡，没空余给 talker/code2wav；non-disagg 只用 5 卡，剩 3 卡空。

## 5. 常用参数速查

`run_server_disagg.sh` / `run_all_stages_disagg.sh` / `run_all_stages_nondisagg.sh`：

| flag | 默认 | 说明 |
|---|---|---|
| `--model` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 模型 id/path |
| `--port` | `8091` | API server 端口（stage 0） |
| `--stage-configs` | `qwen3_omni_a5000.yaml`（`run_server_disagg.sh`） / 各自变体 yaml（多 stage 启动器） | stage configs yaml |
| `--master-port` | `26000` | Omni orchestrator 端口 |
| `--debug` / `--debug-port` | off / 5678 | debugpy listen + wait-for-client |
| `--debug-stages` | `0,1,2,3` (PD) / `0,1,2` (non) | 仅多 stage 启动器 |
| `--debug-ports` | `5678,5679,5680,5681` (PD) / `5678,5679,5680` (non) | 仅多 stage 启动器 |
