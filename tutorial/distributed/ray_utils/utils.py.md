# `utils.py` — Ray 工具函数

## 文件概述

该文件提供了与 Ray 分布式框架集成的全部工具函数，覆盖环境检测、集群管理、Placement Group、Actor 生命周期和内存管理等方面。所有 Ray 依赖都是可选的。

## 关键代码解析

### 1. Ray 可选导入

```python
try:
    import ray
    from ray.util.queue import Queue as RayQueue
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False
```

### 2. is_ray_initialized — Ray 环境检测

```python
def is_ray_initialized():
    if RAY_AVAILABLE:
        if ray.is_initialized():
            return True
    # 回退：检查 RAY_RAYLET_PID 环境变量
    if "RAY_RAYLET_PID" in os.environ:
        return True
    return False
```

双重检测：API 检查 + 环境变量回退（处理 Ray Worker 进程中 `ray` 模块可能未 import 的情况）。

### 3. maybe_disable_pin_memory_for_ray — 大分配保护

```python
@contextmanager
def maybe_disable_pin_memory_for_ray(obj, size_bytes, threshold=32 * 1024 * 1024):
    """在 Ray 环境中大分配时临时禁用 pin_memory"""
    if in_ray and is_large and is_pinned:
        obj.pin_memory = False
    try:
        yield
    finally:
        if should_disable:
            obj.pin_memory = old_pin  # 恢复
```

背景：Ray Worker 通常有较低的 `ulimit -l`（锁定内存限制），大块 pinned memory 分配会导致 OS 错误。此上下文管理器在分配期间临时关闭 pinning。

### 4. calculate_total_bytes — 计算分配大小

```python
def calculate_total_bytes(size_args, dtype):
    """处理嵌套 tuple 的 tensor 大小计算"""
    num_elements = 1
    for s in size_args:
        if isinstance(s, (tuple, list)):
            for inner in s:
                num_elements *= inner
        else:
            num_elements *= s
    element_size = torch.tensor([], dtype=dtype).element_size()
    return num_elements * element_size
```

### 5. 集群与 Placement Group 管理

```python
def initialize_ray_cluster(address=None):
    """初始化 Ray 集群，传递 PYTHONPATH 给 Worker"""
    runtime_env = {"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}}
    ray.init(address=address, ignore_reinit_error=True, runtime_env=runtime_env)

def create_placement_group(number_of_stages, address=None, strategy="PACK"):
    """创建 Placement Group：每个 Stage 1 GPU + 1 CPU"""
    bundles = [{"GPU": 1.0, "CPU": 1.0} for _ in range(number_of_stages)]
    pg = ray.util.placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg
```

### 6. Actor 管理

```python
def start_ray_actor(worker_entry_fn, placement_group, placement_group_bundle_index, *args, **kwargs):
    """启动 Ray Actor 并执行 worker 入口函数"""
    @ray.remote(num_gpus=1)
    class OmniStageRayWorker:
        def run(self, func, *args, **kwargs):
            return func(*args, **kwargs)

    worker_actor = OmniStageRayWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=placement_group_bundle_index
        ),
    ).remote()
    task_ref = worker_actor.run.remote(worker_entry_fn, *args, **kwargs)
    return worker_actor, task_ref

def is_ray_task_alive(task_ref):
    """检查 Ray 任务是否仍在运行"""
    ready, _ = ray.wait([task_ref], timeout=0)
    return not bool(ready)

def get_ray_task_error(task_ref):
    """获取 Ray 任务的错误（如果有）"""
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `is_ray_initialized()` | function | 检测 Ray 环境 |
| `calculate_total_bytes()` | function | 计算 tensor 总字节数 |
| `maybe_disable_pin_memory_for_ray()` | context manager | Ray 下大分配保护 |
| `initialize_ray_cluster()` | function | 初始化 Ray 集群 |
| `create_placement_group()` | function | 创建 Placement Group |
| `remove_placement_group()` | function | 删除 Placement Group |
| `start_ray_actor()` | function | 启动 Ray Actor |
| `kill_ray_actor()` | function | 终止 Ray Actor |
| `is_ray_task_alive()` | function | 检查任务存活状态 |
| `get_ray_task_error()` | function | 获取任务错误 |
| `get_ray_queue_class()` | function | 获取 Ray Queue 构造器 |
| `try_close_ray()` | function | 清理 Ray 资源 |

## 与其他模块的关系

- 被 Orchestrator 的 Ray 后端模式调用
- `maybe_disable_pin_memory_for_ray` 被模型运行时使用
- `is_ray_initialized` 被 `initialization.py` 间接使用（判断是否禁用 SHM）

## 总结

`utils.py` 是 Ray 分布式后端的完整工具集。它处理了 Ray 环境的检测、集群初始化、资源调度（Placement Group）、Actor 生命周期管理以及 Ray Worker 特有的内存限制问题。所有 Ray 依赖都是可选的，确保非 Ray 部署不受影响。
