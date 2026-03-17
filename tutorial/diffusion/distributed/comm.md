# `comm.py` -- 底层通信原语

## 文件概述

`comm.py` 提供了序列并行所需的底层通信原语，包括 4D 和 5D 张量的 All-to-All 操作，以及 Ring Attention 所需的点对点环形通信工具。这些函数改编自 `long-context-attention` 项目，是 Ulysses 注意力和 Ring 注意力的通信基础。

## 关键代码解析

### all_to_all_4D -- 4D 张量 All-to-All

```python
def all_to_all_4D(input, scatter_idx=2, gather_idx=1, group=None, use_sync=False):
    """
    4D 张量的 All-to-All 通信，用于 QKV 的序列/注意力头维度重分布。

    scatter_idx=2, gather_idx=1 时:
      输入: (bs, seqlen/P, hc, hs) -> 输出: (bs, seqlen, hc/P, hs)
      效果: 从序列维度分片 -> 注意力头维度分片

    scatter_idx=1, gather_idx=2 时:
      输入: (bs, seqlen, hc/P, hs) -> 输出: (bs, seqlen/P, hc, hs)
      效果: 从注意力头维度分片 -> 序列维度分片
    """
```

核心变换步骤（以 `scatter_idx=2, gather_idx=1` 为例）：
1. `(bs, seqlen/P, hc, hs)` reshape -> `(bs, seqlen/P, P, hc/P, hs)`
2. transpose -> `(P, seqlen/P, bs, hc/P, hs)`
3. all_to_all -> `(P, seqlen/P, bs, hc/P, hs)` 重分布
4. reshape + transpose -> `(bs, seqlen, hc/P, hs)`

### all_to_all_5D -- 5D 张量 All-to-All

```python
def all_to_all_5D(input, scatter_idx=3, gather_idx=1, group=None, use_sync=False):
    """
    5D 张量的 All-to-All 通信，用于包含 QKV 维度的张量。

    scatter_idx=3, gather_idx=1 时:
      输入: (bs, seqlen/P, 3, hc, hs) -> 输出: (bs, seqlen, 3, hc/P, hs)
    """
```

与 4D 版本类似，但额外处理了 QKV 合并维度（dim=2, 大小为 3）。

### SeqAllToAll4D / SeqAllToAll5D -- 自动微分包装

```python
class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, scatter_idx, gather_idx, use_sync=False):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)
```

将 All-to-All 操作包装为 PyTorch 自动微分函数，保存上下文以支持反向传播。

### RingComm -- 环形通信工具

```python
class RingComm:
    """Ring Attention P2P 通信工具。"""

    def __init__(self, process_group):
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

    def send_recv(self, to_send, recv_tensor=None):
        """异步发送到下一个 rank，同时从上一个 rank 接收。"""
        res = recv_tensor or torch.empty_like(to_send)
        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
```

`RingComm` 实现了环形拓扑的异步点对点通信。使用模式为：
1. 调用 `send_recv()` 注册发送/接收操作
2. 调用 `commit()` 批量发起异步操作
3. 调用 `wait()` 等待完成

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `all_to_all_4D()` | 函数 | 4D 张量的 All-to-All 通信 |
| `all_to_all_5D()` | 函数 | 5D 张量的 All-to-All 通信（含 QKV 维度） |
| `SeqAllToAll4D` | autograd.Function | 4D All-to-All 的自动微分包装 |
| `SeqAllToAll5D` | autograd.Function | 5D All-to-All 的自动微分包装 |
| `RingComm` | 类 | Ring Attention 的环形 P2P 通信工具 |

## 与其他模块的关系

- **parallel_state.py**: 使用 `get_sp_group()` 获取序列并行进程组
- **sp_sharding.py**: 高层分片操作内部可能使用这些通信原语
- **注意力模块**: Ulysses 注意力使用 All-to-All，Ring 注意力使用 RingComm

## 总结

`comm.py` 提供了序列并行的两种通信模式的底层实现：
- **Ulysses (All-to-All)**: 将序列维度分片与注意力头维度分片之间进行重分布，适合带宽充足的场景
- **Ring (P2P)**: 通过环形拓扑传递 KV 块，适合长序列和内存受限的场景

这些原语是 vllm-omni 混合 Ulysses-Ring 序列并行策略的基础设施。
