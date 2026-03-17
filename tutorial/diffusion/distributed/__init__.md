# `__init__.py` -- 分布式模块入口与公共接口导出

## 文件概述

`distributed/__init__.py` 是 vllm-omni 扩散模型分布式推理模块的入口文件。它从各子模块中导入并统一暴露分布式相关的配置类、计划类型、分片工具和 HSDP 支持等公共接口。

## 关键代码解析

```python
from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig, apply_hsdp_to_model
from vllm_omni.diffusion.distributed.parallel_state import (
    get_fs_group, get_fully_shard_rank, get_fully_shard_world_size,
)
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelConfig, SequenceParallelInput, SequenceParallelModelPlan,
    SequenceParallelOutput, SequenceParallelPartialInput,
    get_sp_plan_from_model, validate_sp_plan,
)
from vllm_omni.diffusion.distributed.sp_sharding import (
    ShardingValidator, get_sharding_validator, sp_gather, sp_shard, sp_shard_with_padding,
)
```

导出的接口按功能分类：

| 类别 | 组件 |
|------|------|
| 配置 | `SequenceParallelConfig` |
| 计划类型 | `SequenceParallelInput`, `SequenceParallelOutput`, `SequenceParallelPartialInput`, `SequenceParallelModelPlan` |
| 验证 | `validate_sp_plan`, `get_sp_plan_from_model`, `ShardingValidator`, `get_sharding_validator` |
| 分片工具 | `sp_shard`, `sp_gather`, `sp_shard_with_padding` |
| HSDP | `HSDPInferenceConfig`, `apply_hsdp_to_model` |
| FS 工具 | `get_fs_group`, `get_fully_shard_rank`, `get_fully_shard_world_size` |

## 核心类/函数

| 名称 | 来源 | 说明 |
|------|------|------|
| `SequenceParallelConfig` | sp_plan.py | 序列并行配置 |
| `sp_shard` / `sp_gather` | sp_sharding.py | 序列并行分片/聚合工具函数 |
| `HSDPInferenceConfig` | hsdp.py | HSDP 推理配置 |
| `apply_hsdp_to_model` | hsdp.py | 将 HSDP 分片应用到模型 |

## 与其他模块的关系

- **parallel_state.py**: 提供全局并行状态管理
- **sp_plan.py**: 序列并行计划定义
- **sp_sharding.py**: 底层分片操作
- **hsdp.py**: HSDP 模型分片
- **cfg_parallel.py**: CFG 并行支持（未在 `__init__` 中导出，按需导入）
- **comm.py**: 底层通信原语（未导出）
- **group_coordinator.py**: 进程组协调器（未导出）

## 总结

该入口文件精心筛选了公共 API，将序列并行、HSDP 和 Fully Shard 相关的常用工具统一导出。底层的通信原语（comm.py）、进程组管理（group_coordinator.py）和 CFG 并行（cfg_parallel.py）等模块未被导出，表明它们属于内部实现细节。
