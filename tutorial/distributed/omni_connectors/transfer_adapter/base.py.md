# `base.py` — 传输适配器基类

## 文件概述

该文件定义了 `OmniTransferAdapterBase`，为分块传输适配器提供基础框架：后台接收/保存循环线程和请求队列管理。子类只需实现具体的数据处理逻辑。

## 关键代码解析

### 初始化与后台线程

```python
class OmniTransferAdapterBase:
    def __init__(self, config):
        # 接收侧队列
        self._pending_load_reqs = deque()     # 等待轮询的请求
        self._finished_load_reqs = set()      # 已成功获取数据的请求

        # 发送侧队列
        self._pending_save_reqs = deque()     # 等待发送的请求
        self._finished_save_reqs = set()      # 已成功发送的请求

        self.stop_event = threading.Event()

        # 启动后台线程
        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()
        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()
```

### 接收循环

```python
def recv_loop(self):
    while not self.stop_event.is_set():
        while self._pending_load_reqs:
            request = self._pending_load_reqs.popleft()
            is_success = self._poll_single_request(request)
            if not is_success:
                self._pending_load_reqs.append(request)  # 放回队尾重试
        time.sleep(0.001)  # 1ms 轮询间隔
```

### 抽象方法

```python
def _poll_single_request(self, *args, **kwargs):
    """子类实现：轮询连接器获取单个请求的数据"""
    raise NotImplementedError

def _send_single_request(self, *args, **kwargs):
    """子类实现：通过连接器发送单个请求的数据"""
    raise NotImplementedError

def load_async(self, *args, **kwargs):
    """子类实现：注册请求以加载数据"""
    raise NotImplementedError

def save_async(self, *args, **kwargs):
    """子类实现：提交数据以保存"""
    raise NotImplementedError
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniTransferAdapterBase` | class | 传输适配器基类 |
| `recv_loop()` | method | 后台接收循环 |
| `save_loop()` | method | 后台发送循环 |
| `_poll_single_request()` | method (abstract) | 轮询单个请求的数据 |
| `_send_single_request()` | method (abstract) | 发送单个请求的数据 |
| `load_async()` / `save_async()` | method (abstract) | 异步加载/保存接口 |

## 与其他模块的关系

- 被 `OmniChunkTransferAdapter` 继承
- 不直接使用连接器，由子类在 `_poll_single_request` / `_send_single_request` 中调用

## 总结

`OmniTransferAdapterBase` 提供了生产者-消费者模式的基础框架，通过两个后台守护线程分别处理接收和发送任务。子类只需关注具体的数据处理逻辑，无需管理线程和队列。
