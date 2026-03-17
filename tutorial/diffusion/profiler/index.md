# profiler/ — 性能分析器子模块

## 模块概述

`profiler/` 子模块提供了扩散模型推理的性能分析基础设施。通过抽象基类和默认的 Torch Profiler 实现，支持分布式环境下的 per-rank trace 收集和压缩。

## 架构设计

```
ProfilerBase (抽象基类)
  └── TorchProfiler (torch.profiler 实现)
        ├── start() → 启动端到端连续录制
        ├── stop() → 停止、导出 Chrome trace、后台 gzip
        └── step() → 推进分析步骤

CurrentProfiler = TorchProfiler (可切换的全局 profiler)
```

## 文件索引

| 文件 | 说明 |
|------|------|
| [`__init__.py`](__init__.md) | 包入口，设置默认 profiler |
| [`base.py`](base.md) | Profiler 抽象基类 |
| [`torch_profiler.py`](torch_profiler.md) | 基于 torch.profiler 的实现 |

## 核心设计

- **抽象接口**：`ProfilerBase` 定义 start/stop/step/is_active 标准接口
- **端到端录制**：使用超长 `active` 窗口（100000 步），在 stop 时统一导出
- **后台压缩**：通过 `subprocess.Popen` 启动 gzip 进程，避免阻塞 worker
- **分布式感知**：per-rank 生成独立的 trace 文件（`*_rank{N}.json.gz`）
- **全功能录制**：记录 CPU/CUDA 活动、张量形状、内存分配、调用栈和 FLOPS
