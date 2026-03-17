# `parallel_state.py` -- 全局分布式并行状态管理

## 文件概述

`parallel_state.py` 是 vllm-omni 分布式系统的核心，管理所有并行维度的进程组。改编自 vLLM 和 xDiT 项目，它负责初始化和维护数据并行（DP）、CFG 并行、序列并行（SP）、流水线并行（PP）、张量并行（TP）、Fully Shard（FS）和 DiT 全局组等多种并行组。

## 关键代码解析

### 全局并行组变量

```python
_WORLD: GroupCoordinator | None = None          # 全局通信组
_SP: SequenceParallelGroupCoordinator | None = None  # 序列并行组
_PP: PipelineGroupCoordinator | None = None      # 流水线并行组
_CFG: GroupCoordinator | None = None             # CFG 并行组
_DP: GroupCoordinator | None = None              # 数据并行组
_FS: GroupCoordinator | None = None              # Fully Shard 组
_DIT: GroupCoordinator | None = None             # DiT 全局组
```

### RankGenerator -- 正交 rank 组生成器

```python
class RankGenerator:
    def __init__(self, tp, sp, pp, cfg, dp, fs=1, order="tp-sp-pp-cfg-dp", rank_offset=0):
        self.world_size = tp * sp * pp * cfg * dp
        self.name_to_size = {"tp": tp, "sp": sp, "pp": pp, "cfg": cfg, "dp": dp, "fs": fs}

    def get_ranks(self, token, independent_ranks=False):
        """根据 token 生成对应的 rank 组。
        token="sp" -> 序列并行组
        token="tp-dp" -> TP-DP 联合组
        """
        if independent_ranks and token == "fs":
            # FS 组独立于主并行层级
            return [[i*fs...(i+1)*fs] for i in range(world_size//fs)]
        mask = self.get_mask(self.order, token)
        return generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
```

`generate_masked_orthogonal_rank_groups` 使用数学方法生成正交的并行组。例如，对于 `tp=2, sp=2, pp=2, cfg=2, dp=2` 的 32 GPU 配置，能正确生成所有维度的进程组划分。

### initialize_model_parallel -- 初始化模型并行

```python
def initialize_model_parallel(
    data_parallel_size=1, cfg_parallel_size=1,
    sequence_parallel_size=None, ulysses_degree=1, ring_degree=1,
    tensor_parallel_size=1, pipeline_parallel_size=1,
    fully_shard_degree=1, hsdp_replicate_size=1,
    enable_expert_parallel=False, backend=None,
):
    # 1. 验证配置
    dit_parallel_size = dp * cfg * sp * pp * tp
    if world_size < dit_parallel_size:
        raise RuntimeError(...)

    # 2. 创建 RankGenerator
    rank_generator = RankGenerator(tp, sp, pp, cfg, dp, fs=fully_shard_degree)

    # 3. 初始化各并行组
    _DP = init_model_parallel_group(rank_generator.get_ranks("dp"), ..., parallel_mode="data")
    _CFG = init_model_parallel_group(rank_generator.get_ranks("cfg"), ..., parallel_mode="classifier_free_guidance")
    _PP = init_model_parallel_group(rank_generator.get_ranks("pp"), ..., parallel_mode="pipeline")

    # 4. 初始化序列并行（含 Ulysses/Ring 子组）
    ulysses_pg, ring_pg = set_seq_parallel_pg(ulysses_degree, ring_degree, ...)
    _SP = init_model_parallel_group(..., parallel_mode="sequence", ulysses_group=ulysses_pg, ring_group=ring_pg)

    # 5. 初始化 TP 和 FS 组
    vllm_parallel_state._TP = init_model_parallel_group(rank_generator.get_ranks("tp"), ...)
    _FS = init_model_parallel_group(rank_generator.get_ranks("fs", independent_ranks=True), ...)
```

### set_seq_parallel_pg -- 序列并行子组初始化

```python
def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size, ...):
    """
    初始化 Ulysses 和 Ring 子进程组。

    use_ulysses_low=True 时:
      Ulysses 组是连续块: [0,1], [2,3], ...
      Ring 组是跨步的: [0,2], [1,3], ...

    use_ulysses_low=False 时:
      Ring 组是连续块: [0,1], [2,3], ...
      Ulysses 组是跨步的: [0,2], [1,3], ...
    """
```

### 查询函数

```python
# 序列并行
def get_sp_group() -> SequenceParallelGroupCoordinator: ...
def get_sequence_parallel_world_size(): ...
def get_sequence_parallel_rank(): ...
def get_ulysses_parallel_world_size(): ...
def get_ring_parallel_world_size(): ...

# 流水线并行
def get_pp_group() -> PipelineGroupCoordinator: ...
def get_pipeline_parallel_world_size(): ...
def is_pipeline_first_stage(): ...

# CFG 并行
def get_cfg_group() -> GroupCoordinator: ...
def get_classifier_free_guidance_world_size(): ...
def get_classifier_free_guidance_rank(): ...

# 数据并行
def get_dp_group() -> GroupCoordinator: ...
def get_data_parallel_world_size(): ...

# Fully Shard
def get_fs_group() -> GroupCoordinator: ...
def get_fully_shard_world_size(): ...
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `RankGenerator` | 类 | 正交 rank 组生成器 |
| `generate_masked_orthogonal_rank_groups()` | 函数 | 生成掩码正交并行组 |
| `init_distributed_environment()` | 函数 | 初始化分布式环境 |
| `initialize_model_parallel()` | 函数 | 初始化所有模型并行组 |
| `set_seq_parallel_pg()` | 函数 | 初始化 Ulysses/Ring 序列并行子组 |
| `destroy_model_parallel()` | 函数 | 销毁所有并行组 |
| 查询函数系列 | 函数 | 获取各维度的组、rank、world_size 等 |

## 与其他模块的关系

- **group_coordinator.py**: 使用三种协调器类
- **cfg_parallel.py**: 调用 `get_cfg_group()` 等函数
- **sp_sharding.py**: 调用 `get_sp_group()` 等函数
- **hsdp.py**: 调用 `get_fs_group()` 和 `get_world_group()`
- **teacache/hook.py**: 调用 CFG 并行查询函数
- **vllm.distributed.parallel_state**: 共享 TP 和 EP 组

## 总结

`parallel_state.py` 是整个分布式框架的枢纽，管理着六个并行维度的进程组（DP、CFG、SP、PP、TP、FS）。`RankGenerator` 和 `generate_masked_orthogonal_rank_groups` 使用数学方法确保不同并行维度的组正确正交。`initialize_model_parallel` 函数是一站式初始化入口，处理了所有边界情况（如独立 HSDP 模式、混合 Ulysses-Ring 序列并行等）。
