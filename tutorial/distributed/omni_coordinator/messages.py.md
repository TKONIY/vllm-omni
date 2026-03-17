# `messages.py` — 协调器消息数据类

## 文件概述

该文件定义了协调器系统中使用的所有消息数据类和状态枚举，是协调器通信的数据契约。

## 关键代码解析

### StageStatus — 阶段状态枚举

```python
class StageStatus(str, Enum):
    UP = "up"       # 实例就绪，可用
    DOWN = "down"   # 实例优雅关闭
    ERROR = "error" # 实例遇到错误或心跳超时
```

### InstanceEvent — Stage → Coordinator 事件

```python
@dataclass
class InstanceEvent:
    """Stage 发送给 Coordinator 的事件"""
    input_addr: str       # Stage 实例的输入 ZMQ 地址
    output_addr: str      # Stage 实例的输出 ZMQ 地址
    stage_id: int         # 阶段 ID
    event_type: str       # "update" | "heartbeat"
    status: StageStatus   # 当前状态
    queue_length: int     # 当前队列长度
```

### InstanceInfo — 实例元数据

```python
@dataclass
class InstanceInfo:
    """存储在 Coordinator 注册表中的实例信息"""
    input_addr: str
    output_addr: str
    stage_id: int
    status: StageStatus
    queue_length: int
    last_heartbeat: float   # 最后心跳时间戳
    registered_at: float    # 注册时间戳
```

### InstanceList — 实例列表容器

```python
@dataclass
class InstanceList:
    """Coordinator 发布给 Hub 的实例列表"""
    instances: list[InstanceInfo]
    timestamp: float  # 列表更新时间
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `StageStatus` | Enum | 阶段实例状态 (UP/DOWN/ERROR) |
| `InstanceEvent` | dataclass | Stage → Coordinator 的事件载荷 |
| `InstanceInfo` | dataclass | 单个实例的完整元数据 |
| `InstanceList` | dataclass | 实例列表更新容器 |

## 与其他模块的关系

- 被 `OmniCoordinator`、`OmniCoordClientForStage`、`OmniCoordClientForHub` 使用
- 被 `LoadBalancer` 使用（`InstanceInfo` 列表）
- 通过 JSON 序列化在 ZMQ 上传输

## 总结

四个数据类定义了协调器系统的完整通信协议：状态枚举、事件消息、实例信息和实例列表。
