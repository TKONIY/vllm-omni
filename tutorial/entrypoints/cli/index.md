# cli/ 子模块概述

## 模块简介

`cli/` 子模块实现了 vLLM-Omni 的命令行界面，通过拦截 vLLM 的 `vllm` CLI 命令并在检测到 `--omni` 标志时接管执行。它提供了模型服务启动（`serve`）和性能基准测试（`bench`）两个主要子命令。

## 架构图

```
                    ┌────────────────────┐
                    │  vllm CLI 命令      │
                    │  (vllm serve ...)   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  main.py           │
                    │  检测 --omni 标志   │
                    └───┬────────────┬───┘
                        │            │
          ┌─────────────▼──┐   ┌────▼──────────────┐
          │  没有 --omni    │   │  有 --omni         │
          │  → vLLM 原生   │   │  → Omni CLI 路由   │
          └────────────────┘   └───┬────────────┬───┘
                                   │            │
                       ┌───────────▼──┐   ┌────▼──────────┐
                       │  serve.py     │   │  benchmark/   │
                       │  OmniServe    │   │  基准测试     │
                       │  Command      │   │  子命令       │
                       └───────────────┘   └──────────────┘
```

## 使用示例

```bash
# 启动 Omni LLM 服务
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091

# 启动扩散模型服务
vllm serve Qwen/Qwen-Image --omni --port 8091

# 运行服务基准测试
vllm bench serve --omni [options]
```

## 文件索引

- [\_\_init\_\_.py](./\_\_init\_\_.py.md) — CLI 包初始化
- [logo.py](./logo.py.md) — 启动 Logo 显示
- [main.py](./main.py.md) — CLI 入口点
- [serve.py](./serve.py.md) — serve 子命令实现
- [benchmark/](./benchmark/index.md) — 基准测试子模块
