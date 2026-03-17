# `scheduler.py` — 请求调度器

## 文件概述

`scheduler.py` 实现了 `Scheduler` 类，负责在主进程与 Worker 进程之间调度扩散请求。它通过 vLLM 的 `MessageQueue`（基于共享内存的广播队列）向所有 worker 广播请求，并从 Rank 0 的结果队列接收推理结果。

## 关键代码解析

### Scheduler 初始化

```python
class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        self.num_workers = od_config.num_gpus
        self._lock = threading.Lock()
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )
        self.result_mq = None
```

创建一个广播消息队列，所有 worker 都作为读取者。结果队列由 worker 端创建后通过 handle 传递给 scheduler。

### add_req — 提交请求并等待结果

```python
def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
    with self._lock:
        rpc_request = {
            "type": "rpc",
            "method": "generate",
            "args": (request,),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
        }
        self.mq.enqueue(rpc_request)
        output = self.result_mq.dequeue()
        unpack_diffusion_output_shm(output)
        return output
```

关键特点：
1. 使用线程锁保证同步访问
2. 将请求封装为 RPC 消息，广播到所有 worker
3. 从结果队列同步等待结果
4. 对结果执行 SHM 解包（恢复共享内存中的张量）

### 消息传递架构

```
Scheduler ─── [MessageQueue(广播)] ──→ Worker 0, Worker 1, ..., Worker N
                                         │
Worker 0 ─── [ResultQueue] ──→ Scheduler  │
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Scheduler` | 类 | 请求调度器，管理请求广播和结果接收 |
| `Scheduler.initialize` | 方法 | 初始化消息队列和线程锁 |
| `Scheduler.add_req` | 方法 | 提交请求并同步等待结果 |
| `Scheduler.initialize_result_queue` | 方法 | 从 worker 提供的 handle 初始化结果队列 |

## 与其他模块的关系

- 被 `executor/multiproc_executor.py` 的 `MultiprocDiffusionExecutor` 创建和管理
- 使用 `ipc.py` 的 `unpack_diffusion_output_shm` 解包大张量
- 使用 `data.py` 中的 `DiffusionOutput` 和 `OmniDiffusionConfig`
- 依赖 vLLM 的 `MessageQueue`（基于共享内存的广播机制）

## 总结

`Scheduler` 是多进程执行架构中的通信枢纽。它通过 `MessageQueue` 将请求广播给所有 worker，并通过结果队列从 Rank 0 收集输出。线程锁保证了请求处理的串行化，SHM 解包优化了大张量的传输效率。
