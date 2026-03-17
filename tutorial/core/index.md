# core 模块教程 — 调度器核心

## 模块概述

`core/` 模块包含 vllm-omni 的调度器系统，负责管理请求的调度、KV 缓存分配和跨阶段数据传输。提供两种调度器：
- **OmniARScheduler**：用于自回归（AR）模型阶段
- **OmniGenerationScheduler**：用于生成/扩散模型阶段

## 架构图

```
core/
├── __init__.py                         # 空文件
└── sched/
    ├── __init__.py                     # 导出调度器类
    ├── omni_ar_scheduler.py            # AR 调度器
    ├── omni_generation_scheduler.py    # 生成调度器
    └── output.py                       # 调度器输出数据结构

         ┌──────────────────┐
         │  VLLMScheduler    │  ← vLLM 基类
         └────────┬─────────┘
                  │ 继承
        ┌─────────┴──────────┐
        │                    │
        ▼                    ▼
┌───────────────┐  ┌────────────────────┐
│ OmniARScheduler│  │OmniGenerationScheduler│
│               │  │                    │
│ - KV 缓存传输 │  │ - 一步完成推理      │
│ - 特殊 token  │  │ - 异步分块          │
│   触发传输    │  │ - 扩散模型快速路径  │
└───────┬───────┘  └────────┬───────────┘
        │                    │
        ▼                    ▼
┌────────────────────────────────────┐
│  OmniSchedulerOutput               │
│  ├─ scheduled_new_reqs             │
│  │   (OmniNewRequestData)          │
│  ├─ scheduled_cached_reqs          │
│  │   (OmniCachedRequestData)       │
│  └─ finished_requests_needing_     │
│      kv_transfer                   │
└────────────────────────────────────┘
```

## 模块文档索引

| 文件 | 说明 |
|------|------|
| [sched/__init__.py.md](./sched/__init__.py.md) | 调度器模块入口 |
| [sched/omni_ar_scheduler.py.md](./sched/omni_ar_scheduler.py.md) | AR 自回归调度器 |
| [sched/omni_generation_scheduler.py.md](./sched/omni_generation_scheduler.py.md) | 生成/扩散调度器 |
| [sched/output.py.md](./sched/output.py.md) | 调度器输出数据结构 |
