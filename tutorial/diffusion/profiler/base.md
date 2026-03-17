# `base.py` — Profiler 抽象基类

## 文件概述

`base.py` 定义了 `ProfilerBase` 抽象基类，规定了所有扩散 profiler 必须实现的接口。它为性能分析提供了统一的 start/stop/step 生命周期管理。

## 关键代码解析

### ProfilerBase 抽象接口

```python
class ProfilerBase(ABC):
    @abstractmethod
    def start(self, trace_path_template: str) -> str:
        """开始分析，返回 trace 文件路径"""
        pass

    @abstractmethod
    def stop(self) -> str | None:
        """停止分析，返回保存的 trace 文件路径"""
        pass

    @abstractmethod
    def get_step_context(self):
        """返回推进一步的上下文管理器（不活跃时为 nullcontext）"""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """返回是否正在分析"""
        pass

    @classmethod
    def _get_rank(cls) -> int:
        return int(os.getenv("RANK", "0"))
```

接口设计：
- `start`：接收路径模板（不含 rank 和扩展名），由实现类添加后缀
- `stop`：停止并导出 trace，返回文件路径
- `get_step_context`：用于在推理步中标记步骤边界
- `_get_rank`：从环境变量获取当前 rank，用于生成 per-rank 的 trace 文件名

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ProfilerBase` | ABC | Profiler 抽象基类 |
| `start` | 抽象方法 | 开始性能分析 |
| `stop` | 抽象方法 | 停止并导出结果 |
| `is_active` | 抽象方法 | 查询分析状态 |
| `_get_rank` | 类方法 | 获取当前进程 rank |

## 与其他模块的关系

- 被 `profiler/torch_profiler.py` 的 `TorchProfiler` 继承实现
- 被 `diffusion_engine.py` 的 start/stop_profile 间接使用

## 总结

`ProfilerBase` 定义了 profiler 的标准接口，支持 start/stop 生命周期管理和分布式 rank 感知。抽象设计使得可以轻松替换不同的 profiler 实现（如 NVIDIA Nsight、自定义 profiler 等）。
