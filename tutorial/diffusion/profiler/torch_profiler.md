# `torch_profiler.py` — Torch Profiler 实现

## 文件概述

`torch_profiler.py` 实现了基于 `torch.profiler` 的性能分析器 `TorchProfiler`。它以端到端连续录制模式运行，使用 `on_trace_ready` 回调处理 trace 导出，并通过后台子进程进行 gzip 压缩以避免阻塞 worker 循环。

## 关键代码解析

### start — 启动分析

```python
class TorchProfiler(ProfilerBase):
    _profiler: profile | None = None
    _trace_template: str = ""

    @classmethod
    def start(cls, trace_path_template: str) -> str:
        # 1. 清理已有 profiler
        if cls._profiler is not None:
            cls._profiler.stop()

        # 2. 定义 trace 导出回调
        def trace_handler(p):
            p.export_chrome_trace(json_file)
            subprocess.Popen(["gzip", "-f", json_file])  # 后台压缩

        # 3. 初始化 profiler
        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=100000),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        cls._profiler.start()
        return f"{trace_path_template}_rank{rank}.json.gz"
```

关键配置：
- `active=100000`：极长的活跃窗口，确保在 stop 前持续录制
- `record_shapes=True`：记录张量形状
- `profile_memory=True`：记录内存分配
- `with_stack=True`：记录调用栈
- `with_flops=True`：计算 FLOPS

### stop — 停止分析

```python
@classmethod
def stop(cls) -> dict | None:
    if cls._profiler is None:
        return None
    cls._profiler.stop()  # 触发 trace_handler
    cls._profiler = None
    return {"trace": gz_path, "table": None}
```

`stop()` 调用时会同步触发 `trace_handler`，导出 Chrome trace 文件并启动后台 gzip 压缩。返回预期的 `.json.gz` 路径。

### 类级状态管理

所有方法都是 `@classmethod`，状态存储在类变量中：

```python
_profiler: profile | None = None  # 当前 profiler 实例
_trace_template: str = ""          # trace 路径模板
```

这意味着每个进程只有一个全局 profiler 实例。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `TorchProfiler` | 类 | 基于 torch.profiler 的分析器实现 |
| `start` | 类方法 | 启动端到端连续录制 |
| `stop` | 类方法 | 停止录制，导出 trace 并压缩 |
| `step` | 类方法 | 推进一个分析步骤 |
| `is_active` | 类方法 | 检查是否正在分析 |
| `get_step_context` | 类方法 | 返回 nullcontext（当前不使用步进上下文） |

## 与其他模块的关系

- 继承 `profiler/base.py` 的 `ProfilerBase`
- 通过 `profiler/__init__.py` 设置为默认 profiler (`CurrentProfiler`)
- 被 `worker/diffusion_worker.py` 的 `DiffusionWorker.start_profile/stop_profile` 调用
- 被 `diffusion_engine.py` 通过 `collective_rpc` 间接触发

## 总结

`TorchProfiler` 使用 `torch.profiler` 进行端到端的性能分析，输出标准的 Chrome trace 格式。通过后台 gzip 压缩优化了 trace 文件大小，避免了压缩操作阻塞 worker。类级别的状态管理确保了每个进程全局唯一的 profiler 实例。
