# `load_balancer.py` — 负载均衡器

## 文件概述

该文件定义了负载均衡的抽象基类和一个随机均衡策略实现，用于在多个 Stage 实例之间路由任务。

## 关键代码解析

### Task — 任务类型

```python
class Task(TypedDict, total=False):
    request_id: str
    engine_inputs: Any
    sampling_params: Any
```

与 `AsyncOmni` 中 `stage.submit(task)` 使用的字典结构对应。

### LoadBalancingPolicy — 策略枚举

```python
class LoadBalancingPolicy(str, Enum):
    RANDOM = "random"
```

当前仅实现了随机策略，预留了扩展接口（如 round-robin、least-connections）。

### LoadBalancer — 抽象基类

```python
class LoadBalancer(ABC):
    @abstractmethod
    def select(self, task: Task, instances: list[InstanceInfo]) -> int:
        """返回选中实例在列表中的索引"""
```

### RandomBalancer — 随机均衡器

```python
class RandomBalancer(LoadBalancer):
    def select(self, task, instances) -> int:
        if not instances:
            raise ValueError("instances must not be empty")
        return random.randrange(len(instances))
```

忽略任务内容，均匀随机选择。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `Task` | TypedDict | 任务结构定义 |
| `LoadBalancingPolicy` | Enum | 负载均衡策略枚举 |
| `LoadBalancer` | ABC | 负载均衡器基类 |
| `RandomBalancer` | class | 随机负载均衡器 |

## 与其他模块的关系

- 使用 `InstanceInfo` 获取实例列表
- 被 `AsyncOmni` 用于路由任务到特定 Stage 实例

## 总结

简洁的负载均衡框架。当前仅有随机策略，但抽象基类设计使得未来可以方便地添加更智能的策略（如基于队列长度的最小连接策略）。
