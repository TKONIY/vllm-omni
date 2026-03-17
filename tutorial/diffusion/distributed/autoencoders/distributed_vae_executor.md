# `distributed_vae_executor.py` -- 分布式 VAE 执行框架

## 文件概述

`distributed_vae_executor.py` 是分布式 VAE 解码的核心框架，定义了通用的数据结构（`GridSpec`、`TileTask`、`DistributedOperator`）和执行引擎（`DistributedVaeExecutor`）。它实现了 split-exec-merge 三步模式，将 VAE 解码工作分配到多个 GPU 上。

## 关键代码解析

### 数据结构

```python
@dataclass
class GridSpec:
    split_dims: tuple[int, ...]      # 分割维度，如 (2,3) 或 (3,4)
    grid_shape: tuple[int, ...]      # 网格形状，如 (nh, nw)
    tile_spec: dict = field(default_factory=dict)  # 策略特定参数
    output_dtype: torch.dtype | None = None

@dataclass
class TileTask:
    tile_id: int                     # 瓦片唯一 ID
    grid_coord: tuple[int, ...]      # 在网格中的坐标
    tensor: torch.Tensor | list[torch.Tensor]  # 瓦片数据
    workload: int | float = 1        # 工作量（用于负载均衡）

@dataclass
class DistributedOperator:
    split: callable   # 分割函数: z -> (list[TileTask], GridSpec)
    exec: callable    # 执行函数: TileTask -> Tensor
    merge: callable   # 合并函数: dict[coord, Tensor], GridSpec -> Tensor
```

### DistributedVaeExecutor -- 执行引擎

```python
class DistributedVaeExecutor:
    def __init__(self):
        self.group = get_dit_group()
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.parallel_size = 1

    def execute(self, z, operator, broadcast_result=True):
        # 1. 分割为瓦片
        tiletask_list, grid_spec = operator.split(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}

        # 2. 负载均衡分配 + 本地执行
        assigned = self._balance_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank]
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        # 3. 计算全局填充形状
        global_padding_shape = self._compute_global_padding_shape(local_results, z.ndim, z.device)

        # 4. 打包本地瓦片
        local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
            local_results, global_padding_shape, grid_spec, z.device, output_dtype)

        # 5. Gather 瓦片和元数据到 rank0
        meta_gather = self.gather_tensors(local_meta_tensor)
        tile_gather = self.gather_tensors(local_tile_tensor)

        # 6. Rank0 解包并合并
        if self.rank == 0:
            coord_tensor_map = self._unpack_tiles(meta_gather, tile_gather, grid_spec, tid_coord_map)
            result = operator.merge(coord_tensor_map, grid_spec)

        # 7. 可选广播结果
        if broadcast_result:
            result = self._sync_final_result(result, z.ndim, z.device, output_dtype)
```

### _balance_tasks -- 负载均衡

```python
def _balance_tasks(self, task_list, num_rank):
    """贪心负载均衡：将工作量最大的任务优先分配给负载最小的 rank。"""
    workloads = [0] * num_rank
    assigned = [[] for _ in range(num_rank)]
    for task in sorted(task_list, key=lambda t: t.workload, reverse=True):
        r = workloads.index(min(workloads))
        assigned[r].append(task)
        workloads[r] += task.workload
    return assigned
```

### _compute_global_padding_shape -- 全局形状协商

```python
def _compute_global_padding_shape(self, local_results, output_ndim, device):
    """通过 all_reduce(MAX) 获取所有 rank 中最大的瓦片形状。"""
    local_tile_max_dims = [0] * output_ndim
    for _, tile_tensor in local_results:
        for dim_idx, dim_size in enumerate(tile_tensor.shape):
            local_tile_max_dims[dim_idx] = max(local_tile_max_dims[dim_idx], dim_size)
    local_shape_stat = torch.tensor([len(local_results), *local_tile_max_dims], device=device)
    dist.all_reduce(local_shape_stat, op=dist.ReduceOp.MAX, group=self.group)
    return local_shape_stat.tolist()
```

由于不同瓦片可能有不同形状（边缘瓦片较小），需要先协商全局最大形状，再填充为统一大小以便 gather。

### DistributedVaeMixin -- VAE 混入类

```python
class DistributedVaeMixin:
    def init_distributed(self):
        self.distributed_decoder = DistributedVaeExecutor()

    def set_parallel_size(self, parallel_size):
        return self.distributed_decoder.set_parallel_size(parallel_size)

    def is_distributed_enabled(self):
        if self.distributed_decoder.parallel_size <= 1 or not dist.is_initialized():
            return False
        world_size = dist.get_world_size(group=self.distributed_decoder.group)
        pp_size = min(self.distributed_decoder.parallel_size, world_size)
        return pp_size > 1
```

VAE 模型通过继承 `DistributedVaeMixin` 获得分布式能力。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `GridSpec` | dataclass | 网格分割规格 |
| `TileTask` | dataclass | 瓦片任务（含 ID、坐标、数据、工作量） |
| `DistributedOperator` | dataclass | split/exec/merge 三元组 |
| `DistributedVaeExecutor` | 类 | 分布式执行引擎 |
| `DistributedVaeExecutor.execute()` | 方法 | 核心执行流程 |
| `DistributedVaeExecutor._balance_tasks()` | 方法 | 贪心负载均衡 |
| `DistributedVaeMixin` | 类 | VAE 模型的分布式混入 |

## 与其他模块的关系

- **parallel_state.py**: 使用 `get_dit_group()` 获取通信组
- **autoencoder_kl.py**: `DistributedAutoencoderKL` 继承 `DistributedVaeMixin`
- **autoencoder_kl_qwenimage.py**: `DistributedAutoencoderKLQwenImage` 继承 `DistributedVaeMixin`
- **autoencoder_kl_wan.py**: `DistributedAutoencoderKLWan` 继承 `DistributedVaeMixin`

## 总结

`distributed_vae_executor.py` 提供了一个通用的分布式 VAE 解码框架。核心设计理念：
1. **split-exec-merge 模式**: 将问题分解为三个可插拔的步骤
2. **负载均衡**: 贪心算法确保各 GPU 工作量均匀
3. **形状协商**: all_reduce(MAX) 处理异构瓦片形状
4. **混入模式**: `DistributedVaeMixin` 使任何 VAE 都能获得分布式能力

不同 VAE 模型只需实现自己的 split/exec/merge 方法，复杂的分布式通信和协调由框架统一处理。
