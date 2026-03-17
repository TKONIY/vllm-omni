# `group_coordinator.py` -- 进程组协调器

## 文件概述

`group_coordinator.py` 定义了三种进程组协调器类，是 vllm-omni 分布式通信的核心基础设施。改编自 vLLM 的 `parallel_state.py`，这些协调器封装了 PyTorch 的 `ProcessGroup`，提供了丰富的集合通信操作和张量字典广播能力。

## 关键代码解析

### GroupCoordinator -- 基础协调器

```python
class GroupCoordinator:
    rank: int           # 全局 rank
    ranks: list[int]    # 组内所有 rank 列表
    world_size: int     # 组大小
    local_rank: int     # 本地 rank（用于设备分配）
    rank_in_group: int  # 组内 rank
    cpu_group: ProcessGroup    # CPU 通信组（Gloo 后端）
    device_group: ProcessGroup # 设备通信组（NCCL 等）

    def __init__(self, group_ranks, local_rank, torch_distributed_backend):
        for ranks in group_ranks:
            device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.device_group = device_group
                self.cpu_group = cpu_group
```

每个协调器同时维护两个进程组：
- **device_group**: 使用 NCCL 等 GPU 后端，用于张量通信
- **cpu_group**: 使用 Gloo 后端，用于元数据和对象通信

### 核心通信方法

```python
def all_reduce(self, input_, op=ReduceOp.SUM):
    torch.distributed.all_reduce(input_, op=op, group=self.device_group)
    return input_

def all_gather(self, input_, dim=0, separate_tensors=False):
    # 支持在任意维度上聚合，可返回连续张量或独立张量列表
    output_tensor = torch.empty(input_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
    if separate_tensors:
        return [output_tensor.narrow(0, input_.numel() * i, input_.numel()).view_as(input_) ...]
    return output_tensor

def broadcast(self, input_, src=0):
    torch.distributed.broadcast(input_, src=self.ranks[src], group=self.device_group)

def broadcast_tensor_dict(self, tensor_dict=None, src=0):
    # 广播包含张量和非张量的混合字典
    # 1. 分离张量和元数据
    # 2. 通过 CPU 组广播元数据
    # 3. 通过设备组广播张量
```

`broadcast_tensor_dict` 是一个高级方法，能广播包含张量和 Python 对象的混合字典，自动处理序列化和设备放置。

### PipelineGroupCoordinator -- 流水线并行协调器

```python
class PipelineGroupCoordinator(GroupCoordinator):
    def __init__(self, group_ranks, local_rank, torch_distributed_backend):
        # 当 pipeline 并行度为 2 时，创建双向通信组避免死锁
        if len(group_ranks[0]) == 2:
            device_group_0_1 = torch.distributed.new_group(ranks, ...)
            device_group_1_0 = torch.distributed.new_group(ranks, ...)
            self.device_groups = [device_group_0_1, device_group_1_0]
```

扩展了基础协调器，增加了以下能力：
- **双向通信组**: 当流水线并行度为 2 时，创建两个独立的通信组避免通信死锁
- **异步接收缓冲**: 支持预分配接收缓冲区和异步 P2P 通信
- **形状协商**: 在首次通信前自动协商张量形状
- **跳跃连接**: 支持非相邻 rank 之间的 P2P 通信

关键方法：

```python
def pipeline_send(self, tensor, name="latent", segment_idx=-1):
    """同步发送张量到下一个 rank"""

def pipeline_recv(self, idx=-1, name="latent"):
    """同步接收来自前一个 rank 的张量"""

def add_pipeline_recv_task(self, idx=-1, name="latent"):
    """添加异步接收任务到队列"""

def recv_next(self):
    """启动队列中的下一个异步接收"""

def get_pipeline_recv_data(self, idx=-1, name="latent"):
    """获取已完成的异步接收数据"""
```

### SequenceParallelGroupCoordinator -- 序列并行协调器

```python
class SequenceParallelGroupCoordinator(GroupCoordinator):
    def __init__(self, group_ranks, local_rank, torch_distributed_backend, **kwargs):
        super().__init__(group_ranks=group_ranks, ...)
        self.ulysses_group = kwargs.get("ulysses_group")
        self.ring_group = kwargs.get("ring_group")
        self.ulysses_world_size = torch.distributed.get_world_size(self.ulysses_group)
        self.ring_world_size = torch.distributed.get_world_size(self.ring_group)
```

在基础协调器上额外维护 Ulysses 和 Ring 两个子进程组，支持混合序列并行策略。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `GroupCoordinator` | 类 | 基础进程组协调器，提供完整的集合通信 |
| `PipelineGroupCoordinator` | 类 | 流水线并行协调器，支持异步 P2P 和跳跃连接 |
| `SequenceParallelGroupCoordinator` | 类 | 序列并行协调器，维护 Ulysses/Ring 子组 |
| `_split_tensor_dict()` | 函数 | 将混合字典分离为元数据和张量列表 |
| `_update_nested_dict()` | 函数 | 从扁平化键重建嵌套字典 |

## 与其他模块的关系

- **parallel_state.py**: 使用这些协调器类创建全局并行组实例
- **cfg_parallel.py**: 通过 `get_cfg_group()` 获取 `GroupCoordinator` 实例
- **comm.py**: Ring 通信使用底层的 P2P 操作
- **vae_patch_parallel.py**: 使用 `GroupCoordinator` 进行 VAE 分布式解码

## 总结

`group_coordinator.py` 是分布式通信的基石，三种协调器类覆盖了所有并行场景：
- `GroupCoordinator`: 数据并行、CFG 并行、Fully Shard 等简单组
- `PipelineGroupCoordinator`: 流水线并行（含双向通信组防死锁、异步 P2P 等高级功能）
- `SequenceParallelGroupCoordinator`: 序列并行（含 Ulysses/Ring 子组）

这些协调器统一了 CPU 和 GPU 通信的管理，简化了上层并行逻辑的实现。
