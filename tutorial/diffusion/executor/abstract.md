# `abstract.py` — 执行器抽象基类

## 文件概述

`abstract.py` 定义了 `DiffusionExecutor` 抽象基类，规定了所有扩散模型执行器必须实现的接口。同时提供了工厂方法 `get_class`，根据配置动态选择执行器后端（多进程、Ray、自定义等）。

## 关键代码解析

### DiffusionExecutor 抽象基类

```python
class DiffusionExecutor(ABC):
    uses_multiproc: bool = False

    @staticmethod
    def get_class(od_config: OmniDiffusionConfig) -> type["DiffusionExecutor"]:
        distributed_executor_backend = od_config.distributed_executor_backend
        if distributed_executor_backend == "mp":
            from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
            executor_class = MultiprocDiffusionExecutor
        elif distributed_executor_backend == "ray":
            raise NotImplementedError("ray backend is not yet supported.")
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
        # ...
        return executor_class
```

`get_class` 是工厂方法，支持以下后端类型：
- `"mp"`：多进程执行器（默认，当前唯一完整实现）
- `"ray"`：Ray 分布式（预留）
- `"external_launcher"`：外部启动器（预留）
- 自定义 Python 路径字符串：通过 `resolve_obj_by_qualname` 动态加载

### 抽象接口

```python
@abstractmethod
def _init_executor(self) -> None: ...

@abstractmethod
def add_req(self, requests: OmniDiffusionRequest) -> DiffusionOutput: ...

@abstractmethod
def collective_rpc(self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None) -> Any: ...

@abstractmethod
def check_health(self) -> None: ...

@abstractmethod
def shutdown(self) -> None: ...
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `DiffusionExecutor` | ABC | 执行器抽象基类，定义标准接口 |
| `get_class` | 静态方法 | 工厂方法，根据配置返回具体执行器类 |
| `_init_executor` | 抽象方法 | 初始化执行器（启动 worker、建立 IPC） |
| `add_req` | 抽象方法 | 提交扩散请求 |
| `collective_rpc` | 抽象方法 | 向所有 worker 发起 RPC |
| `check_health` | 抽象方法 | 健康检查 |
| `shutdown` | 抽象方法 | 关闭执行器 |

## 与其他模块的关系

- 被 `diffusion_engine.py` 调用 `get_class` 获取执行器类并实例化
- `multiproc_executor.py` 中的 `MultiprocDiffusionExecutor` 继承此基类
- 依赖 `data.py` 的 `OmniDiffusionConfig` 和 `DiffusionOutput`

## 总结

`DiffusionExecutor` 通过抽象基类和工厂方法实现了执行器的可扩展架构。当前默认使用多进程执行器 (`"mp"`)，但架构设计为可以轻松扩展到 Ray 或自定义后端。
