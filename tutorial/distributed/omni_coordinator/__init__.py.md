# `__init__.py` — omni_coordinator 包入口

## 文件概述

该文件从各子模块导入并导出协调器系统的所有公共接口。

## 关键代码解析

```python
from .load_balancer import LoadBalancer, LoadBalancingPolicy, RandomBalancer, Task
from .messages import InstanceEvent, InstanceInfo, InstanceList, StageStatus
from .omni_coord_client_for_hub import OmniCoordClientForHub
from .omni_coord_client_for_stage import OmniCoordClientForStage
from .omni_coordinator import OmniCoordinator
```

## 核心类/函数

| 名称 | 用途 |
|------|------|
| `OmniCoordinator` | 协调器服务 |
| `OmniCoordClientForStage` | Stage 端客户端 |
| `OmniCoordClientForHub` | Hub 端客户端 |
| `StageStatus` | 阶段状态枚举 |
| `InstanceEvent` / `InstanceInfo` / `InstanceList` | 消息数据类 |
| `LoadBalancer` / `RandomBalancer` | 负载均衡器 |
| `Task` | 任务类型定义 |

## 总结

导入文件，统一暴露协调器系统的 API。
