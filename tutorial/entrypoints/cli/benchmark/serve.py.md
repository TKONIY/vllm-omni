# `serve.py` — serve 基准测试子命令

## 文件概述

实现了 `vllm bench serve --omni` 命令，用于测试 vLLM-Omni 在线服务的吞吐量和延迟。继承自 vLLM 标准的 benchmark serve 参数，并扩展了 Omni 特有的指标（如音频首包延迟、实时率）。

## 关键代码解析

```python
class OmniBenchmarkServingSubcommand(OmniBenchmarkSubcommandBase):
    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser):
        add_cli_args(parser)  # 继承 vLLM 标准参数
        # 定制 Omni 特有的参数说明
        for action in parser._actions:
            if action.dest == "percentile_metrics":
                action.help = (
                    "...允许的指标名: ttft, tpot, itl, e2el, audio_ttfp, audio_rtf"
                )

    @staticmethod
    def cmd(args):
        main(args)  # 调用 benchmarks/serve.py 的 main 函数
```

在标准 vLLM benchmark 参数基础上，扩展了以下 Omni 特有指标：
- `audio_ttfp`: 音频首包时间（Time To First Packet）
- `audio_rtf`: 音频实时率（Real-Time Factor）

同时增强了多模态输入配置的帮助说明，支持音频输入模态。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniBenchmarkServingSubcommand` | 类 | serve 基准测试子命令 |
| `add_cli_args()` | 类方法 | 注册 CLI 参数并定制帮助文本 |
| `cmd()` | 静态方法 | 执行基准测试 |

## 与其他模块的关系

- 继承 `OmniBenchmarkSubcommandBase`（`base.py`）
- 复用 `vllm.benchmarks.serve` 的 CLI 参数和核心测试逻辑
- 调用 `vllm_omni.benchmarks.serve.main()` 执行实际测试

## 总结

一个轻量的子命令适配器，在 vLLM 标准 benchmark 基础上添加了 Omni 特有的音频性能指标和多模态输入配置支持。
