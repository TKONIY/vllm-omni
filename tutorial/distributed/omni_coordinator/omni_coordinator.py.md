# `omni_coordinator.py` — 协调器服务

## 文件概述

该文件实现了 `OmniCoordinator`，是多阶段推理的中心协调服务。它通过 ZMQ ROUTER 接收 Stage 实例的事件（注册、更新、心跳），通过 ZMQ PUB 向 Hub 客户端广播活跃实例列表。

## 关键代码解析

### 初始化

```python
class OmniCoordinator:
    def __init__(self, router_zmq_addr, pub_zmq_addr, heartbeat_timeout=30.0):
        self._ctx = zmq.Context()

        # ROUTER socket: 接收 Stage 客户端的 DEALER 消息
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(router_zmq_addr)

        # PUB socket: 广播实例列表给 Hub 客户端
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(pub_zmq_addr)

        self._instances: dict[str, InstanceInfo] = {}  # input_addr → InstanceInfo

        # 启动后台线程
        self._recv_thread   # 接收事件
        self._periodic_thread  # 心跳检查 + 广播节流
```

### 事件处理

```python
def _handle_event(self, event: InstanceEvent):
    input_addr = event.input_addr

    if event.event_type == "heartbeat":
        # 仅更新 last_heartbeat
        # 如果之前是 ERROR 状态，恢复为 UP 并立即广播
        info.last_heartbeat = time()
        if info.status == StageStatus.ERROR:
            info.status = StageStatus.UP
            self._schedule_broadcast(force=True)
        return

    with self._lock:
        if input_addr not in self._instances:
            self._add_new_instance_locked(event)    # 新实例注册
            force_broadcast = True
        elif event.status == StageStatus.DOWN:
            self._remove_instance_locked(event)     # 实例下线
            force_broadcast = True
        else:
            self._update_instance_info_locked(event) # 状态更新

    self._schedule_broadcast(force=force_broadcast)
```

### 广播节流

```python
def _schedule_broadcast(self, force: bool):
    """force=True 立即广播，否则标记 pending 等待周期循环刷新"""
    if force:
        self.publish_instance_list_update()
    else:
        self._pending_broadcast = True

def _periodic_loop(self):
    """周期循环：心跳超时检查 + pending 广播刷新"""
    if self._pending_broadcast:
        self.publish_instance_list_update()
        self._pending_broadcast = False
```

关键设计：队列长度等频繁变化的更新被节流（每 0.1 秒最多一次广播），而实例注册/下线/心跳恢复等关键事件立即广播。

### 心跳超时检查

```python
def _check_heartbeat_timeouts(self):
    now = time()
    for input_addr, info in self._instances.items():
        if info.status == StageStatus.UP and now - info.last_heartbeat > self._heartbeat_timeout:
            info.status = StageStatus.ERROR  # 标记为异常
        elif info.status in (DOWN, ERROR) and now - info.last_heartbeat > 600:
            to_delete.append(input_addr)     # 10 分钟后从注册表删除
```

### 实例列表广播

```python
def publish_instance_list_update(self):
    active_list = self.get_active_instances()  # 仅 UP 状态的实例
    payload = asdict(active_list)
    data = json.dumps(payload).encode("utf-8")
    self._pub.send(data, flags=zmq.NOBLOCK)    # 尽力发送
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniCoordinator` | class | 中心协调器服务 |
| `get_active_instances()` | method | 获取所有 UP 状态的实例列表 |
| `add_new_instance()` | method | 注册新实例 |
| `update_instance_info()` | method | 更新实例信息 |
| `remove_instance()` | method | 标记实例下线 |
| `publish_instance_list_update()` | method | 广播实例列表 |
| `_handle_event()` | method | 事件分发 |
| `_check_heartbeat_timeouts()` | method | 心跳超时检查 |
| `close()` | method | 关闭所有线程和 socket |

## 与其他模块的关系

- 接收 `OmniCoordClientForStage` 发来的 `InstanceEvent`
- 向 `OmniCoordClientForHub` 广播 `InstanceList`
- 使用 `messages.py` 中的数据类

## 总结

`OmniCoordinator` 是分布式多阶段推理的服务发现和健康监控中心。它通过 ZMQ ROUTER/PUB 模式实现了高效的事件驱动架构：ROUTER 接收多个 Stage 的并发事件，PUB 广播给所有订阅的 Hub。广播节流机制和两级超时（30 秒心跳超时 + 10 分钟 GC 清理）确保了系统的高效和健壮。
