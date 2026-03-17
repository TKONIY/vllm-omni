# benchmark/ 子模块概述

## 模块简介

`benchmark/` 子模块为 vLLM-Omni 提供性能基准测试的 CLI 支持。通过 `vllm bench serve --omni` 命令可以测试在线服务的吞吐量和延迟指标。

## 架构图

```
┌─────────────────────────┐
│  vllm bench --omni      │
│  (benchmark/main.py)    │
└───────────┬─────────────┘
            │
  ┌─────────▼──────────┐
  │ OmniBenchmark       │
  │ Subcommand          │
  │ (bench 子命令路由)  │
  └─────────┬──────────┘
            │
  ┌─────────▼──────────────┐
  │ OmniBenchmarkServing   │
  │ Subcommand             │
  │ (serve 基准测试)       │
  │ benchmark/serve.py     │
  └─────────┬──────────────┘
            │
  ┌─────────▼──────────┐
  │ benchmarks/serve.py │
  │ (实际测试逻辑)      │
  └────────────────────┘
```

## 使用示例

```bash
# 运行在线服务基准测试
vllm bench serve --omni --backend vllm \
  --model Qwen/Qwen2.5-Omni-7B \
  --num-prompts 100
```

## 文件索引

- [\_\_init\_\_.py](./\_\_init\_\_.py.md) — 包初始化（空文件）
- [base.py](./base.py.md) — 基准测试子命令基类
- [main.py](./main.py.md) — bench 子命令路由
- [serve.py](./serve.py.md) — serve 基准测试子命令
