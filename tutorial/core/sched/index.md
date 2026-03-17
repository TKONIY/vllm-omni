# sched 子模块索引 — 调度器实现

## 模块概述

`sched/` 子模块包含 vllm-omni 的两种核心调度器实现：`OmniARScheduler`（自回归调度器）和 `OmniGenerationScheduler`（生成调度器），以及调度器输出的数据结构定义。

## 文档列表

| 文件 | 说明 |
|------|------|
| [__init__.py.md](__init__.py.md) | 包初始化与调度器类导出 |
| [omni_ar_scheduler.py.md](omni_ar_scheduler.py.md) | 自回归（AR）调度器 |
| [omni_generation_scheduler.py.md](omni_generation_scheduler.py.md) | 生成调度器 |
| [output.py.md](output.py.md) | 调度器输出数据结构 |
