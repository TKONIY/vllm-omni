# `omni_coord_client_for_stage.py` — Stage 端协调客户端

## 文件概述

该文件实现了 `OmniCoordClientForStage`，供 Stage 实例使用。它通过 ZMQ DEALER socket 向 `OmniCoordinator` 发送注册、状态更新和心跳事件。

## 关键代码解析

### 初始化

```python
class OmniCoordClientForStage:
    def __init__(self, coord_zmq_addr, input_addr, output_addr, stage_id):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.connect(coord_zmq_addr)

        self._status = StageStatus.UP
        self._queue_length = 0
        self._heartbeat_interval = 5.0

        # 发送初始注册事件
        self._send_event("update")

        # 启动心跳线程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
```

### 事件发送（带重连）

```python
def _send_event(self, event_type):
    event = InstanceEvent(
        input_addr=self._input_addr,
        output_addr=self._output_addr,
        stage_id=self._stage_id,
        event_type=event_type,
        status=self._status,
        queue_length=self._queue_length,
    )
    data = json.dumps(asdict(event)).encode("utf-8")

    with self._send_lock:
        try:
            self._socket.send(data, flags=zmq.NOBLOCK)
        except (RuntimeError, zmq.ZMQError):
            if not self._reconnect():
                raise
            self._socket.send(data, flags=zmq.NOBLOCK)
```

发送失败时尝试重连（5 秒重试间隔，持续直到成功或被停止）。

### 状态更新

```python
def update_info(self, status=None, queue_length=None):
    """更新实例信息并通知 Coordinator"""
    if status is not None:
        self._status = status
    if queue_length is not None:
        self._queue_length = queue_length
    self._send_event("update")
```

### 心跳循环

```python
def _heartbeat_loop(self):
    while not self._stop_event.wait(timeout=self._heartbeat_interval):
        self._send_event("heartbeat")
```

每 5 秒发送一次心跳。

### 优雅关闭

```python
def close(self):
    self._stop_event.set()               # 停止心跳
    self._heartbeat_thread.join()
    self._status = StageStatus.DOWN
    self._send_event("update")           # 发送下线通知
    self._socket.close(0)
    self._ctx.term()
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniCoordClientForStage` | class | Stage 端协调客户端 |
| `update_info()` | method | 更新并发送实例状态 |
| `_send_event()` | method | 发送事件（带重连） |
| `_heartbeat_loop()` | method | 心跳后台循环 |
| `_reconnect()` | method | ZMQ socket 重连 |
| `close()` | method | 优雅关闭（发送 DOWN 事件） |

## 与其他模块的关系

- 连接 `OmniCoordinator` 的 ROUTER socket
- 被 Stage Worker 使用，报告实例状态和队列长度
- 使用 `messages.py` 中的 `InstanceEvent`、`StageStatus`

## 总结

`OmniCoordClientForStage` 是 Stage 实例与协调器通信的桥梁。通过 DEALER socket 实现非阻塞发送，5 秒心跳保持存活感知，自动重连确保网络恢复后的连接可用性。优雅关闭时发送 DOWN 通知让协调器及时移除实例。
