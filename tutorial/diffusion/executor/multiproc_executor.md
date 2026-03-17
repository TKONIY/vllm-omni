# `multiproc_executor.py` — 多进程执行器

## 文件概述

`multiproc_executor.py` 实现了 `MultiprocDiffusionExecutor`，是当前默认的扩散模型执行器。它使用 Python `multiprocessing` 启动多个 GPU Worker 进程，通过 `Scheduler` 进行请求广播和结果收集，并使用 `weakref.finalize` 保证资源的安全清理。

## 关键代码解析

### BackgroundResources — 资源清理器

```python
@dataclass
class BackgroundResources:
    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None

    def __call__(self):
        # 向每个 worker 发送 SHUTDOWN_MESSAGE
        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
        self.scheduler.close()
        # 等待进程退出，超时则终止
        for proc in self.processes:
            proc.join(30)
            if proc.is_alive():
                proc.terminate()
```

通过 `weakref.finalize` 绑定到执行器实例，确保即使异常退出也能清理 worker 进程。

### Worker 启动流程

```python
def _launch_workers(self, broadcast_handle):
    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        process = mp.Process(
            target=WorkerProc.worker_main,
            args=(i, od_config, writer, broadcast_handle, ...),
            daemon=True,
        )
        process.start()

    # 等待所有 worker 就绪
    for i, reader in enumerate(scheduler_pipe_readers):
        data = reader.recv()  # {"status": "ready", "result_handle": ...}
```

启动流程：
1. 为每个 GPU 创建一个 `mp.Pipe` 用于初始化同步
2. 以 `spawn` 模式启动 worker 进程
3. Worker 完成初始化后通过 Pipe 回传就绪状态和结果队列 handle
4. Rank 0 的 result_handle 用于初始化 Scheduler 的结果队列

### collective_rpc — 分布式 RPC

```python
def collective_rpc(self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None):
    rpc_request = {
        "type": "rpc", "method": method,
        "args": args, "kwargs": kwargs,
        "output_rank": unique_reply_rank,
    }
    # 获取 scheduler 锁（带超时）
    self.scheduler.mq.enqueue(rpc_request)
    # 收集响应
    for _ in range(num_responses):
        response = self.scheduler.result_mq.dequeue(timeout=dequeue_timeout)
    return responses
```

支持指定 `unique_reply_rank` 仅从特定 rank 获取响应，或从所有 worker 收集响应。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `MultiprocDiffusionExecutor` | 类 | 多进程执行器，管理 worker 生命周期 |
| `BackgroundResources` | dataclass | 资源清理器，通过 finalize 绑定 |
| `_init_executor` | 方法 | 初始化 Scheduler 并启动 worker 进程 |
| `_launch_workers` | 方法 | 启动 worker 进程并等待就绪 |
| `add_req` | 方法 | 通过 Scheduler 提交请求 |
| `collective_rpc` | 方法 | 向 worker 广播 RPC 并收集结果 |
| `check_health` | 方法 | 检查 worker 进程是否存活 |
| `shutdown` | 方法 | 触发资源清理 |

## 与其他模块的关系

- 继承 `executor/abstract.py` 的 `DiffusionExecutor`
- 使用 `scheduler.py` 的 `Scheduler` 进行消息路由
- 启动 `worker/diffusion_worker.py` 中的 `WorkerProc`
- 使用 `data.py` 中的 `SHUTDOWN_MESSAGE` 信号关闭 worker

## 总结

`MultiprocDiffusionExecutor` 是扩散模型的默认执行器实现。它通过 Python multiprocessing 启动 worker 进程，使用共享内存 MessageQueue 进行高效通信，并通过 `weakref.finalize` 保证了进程资源的可靠回收。整体架构为：Executor -> Scheduler -> MessageQueue -> Workers。
