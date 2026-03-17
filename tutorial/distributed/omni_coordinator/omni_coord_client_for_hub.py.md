# `omni_coord_client_for_hub.py` — Hub 端协调客户端

## 文件概述

该文件实现了 `OmniCoordClientForHub`，供 AsyncOmni（推理入口）侧使用。它通过 ZMQ SUB socket 订阅 `OmniCoordinator` 发布的实例列表更新，并在内存中缓存最新值供负载均衡和路由使用。

## 关键代码解析

### 初始化

```python
class OmniCoordClientForHub:
    def __init__(self, coord_zmq_addr: str):
        self._ctx = zmq.Context()
        self._instance_list: InstanceList | None = None
        self._lock = threading.Lock()

        # 后台线程接收更新
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        # 等待初始连接完成（最多 5 秒）
        self._init_done.wait(timeout=5.0)
        if self._init_error:
            raise RuntimeError(...)
```

### 接收循环

```python
def _recv_loop(self):
    sub = self._ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")      # 订阅所有消息
    sub.setsockopt(zmq.RCVTIMEO, 100)       # 100ms 超时
    sub.connect(self._coord_zmq_addr)

    while not self._stop_event.is_set():
        try:
            data = sub.recv()
            payload = json.loads(data.decode("utf-8"))
            inst_list = self._decode_instance_list(payload)
            with self._lock:
                self._instance_list = inst_list
        except zmq.Again:
            continue  # 超时，继续轮询
        except zmq.ZMQError:
            # 连接断开，尝试重连
            sub.close()
            sub = None
            sleep(1.0)
```

包含自动重连逻辑：连接断开后 1 秒重试。

### 查询接口

```python
def get_instance_list(self) -> InstanceList:
    """返回最新缓存的实例列表"""
    with self._lock:
        return self._instance_list or InstanceList(instances=[], timestamp=0.0)

def get_instances_for_stage(self, stage_id: int) -> InstanceList:
    """按 stage_id 过滤实例列表"""
    base = self.get_instance_list()
    filtered = [inst for inst in base.instances if inst.stage_id == stage_id]
    return InstanceList(instances=filtered, timestamp=base.timestamp)
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniCoordClientForHub` | class | Hub 端协调客户端 |
| `get_instance_list()` | method | 获取全部活跃实例 |
| `get_instances_for_stage()` | method | 按阶段 ID 过滤实例 |
| `_recv_loop()` | method | 后台接收循环 |
| `_decode_instance_list()` | method | JSON → InstanceList |
| `close()` | method | 关闭 socket 和线程 |

## 与其他模块的关系

- 连接 `OmniCoordinator` 的 PUB socket
- 被 `AsyncOmni` 使用，配合 `LoadBalancer` 进行负载均衡
- 使用 `messages.py` 中的 `InstanceList`、`InstanceInfo`、`StageStatus`

## 总结

`OmniCoordClientForHub` 是一个轻量级的 SUB 客户端，通过后台线程持续接收协调器广播并缓存最新实例列表。线程安全的缓存设计使得查询不会阻塞。自动重连机制确保了网络抖动时的恢复能力。
