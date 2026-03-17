# metrics 模块索引

本模块提供 vllm-omni 编排器（Orchestrator）层面的性能统计和指标聚合能力，用于在多阶段流水线推理过程中跟踪每个阶段和端到端的性能表现。

## 模块结构

```
metrics/
├── __init__.py    # 包初始化与导出
├── stats.py       # 指标数据结构与编排器聚合器
└── utils.py       # 表格格式化与辅助函数
```

## 文档列表

| 文件 | 说明 |
|------|------|
| [__init__.md](__init__.md) | 包初始化与导出 |
| [stats.md](stats.md) | 编排器指标聚合器 |
| [utils.md](utils.md) | 表格格式化工具 |

## 模块间关系

- `stats.py` 定义了所有指标数据结构和 `OrchestratorAggregator` 聚合器。
- `utils.py` 提供 `_format_table` 等工具函数，被 `stats.py` 用于格式化日志输出。
- 本模块主要被编排器（orchestrator）在多阶段推理流水线中调用。
