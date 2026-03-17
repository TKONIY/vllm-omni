# `mooncake_transfer_engine_connector.py` — Mooncake RDMA 传输引擎连接器

## 文件概述

该文件实现了基于 Mooncake Transfer Engine 的高性能连接器 `MooncakeTransferEngineConnector`，支持 RDMA 和 TCP 协议，具备内存池管理和零拷贝传输能力。这是所有连接器中最复杂的实现，约 600+ 行代码。

## 关键代码解析

### 1. 辅助数据结构

```python
@dataclass
class MooncakeAgentMetadata:
    """ZMQ 握手时交换的元数据"""
    remote_hostname: str
    remote_port: int       # RDMA 端口
    request_id: str
    dst_addrs: list[int]   # 目标内存地址列表
    lengths: list[int]     # 每段数据长度列表
```

### 2. BufferAllocator — 内存池分配器

```python
class BufferAllocator:
    def __init__(self, total_size, alignment=4096):
        self.free_blocks = [(0, total_size)]  # 空闲块列表

    def alloc(self, size) -> int:
        """分配对齐后的内存块，返回偏移量"""
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment
        # 遍历空闲块，找到足够大的块...

    def free(self, offset, size):
        """释放内存块，检测双重释放和内存损坏，自动合并相邻空闲块"""
```

线程安全的简单空闲列表分配器。支持：
- 4KB 对齐分配
- 双重释放检测
- 相邻空闲块自动合并

### 3. ManagedBuffer — 托管缓冲区

```python
class ManagedBuffer:
    """全局内存池中的临时视图，必须在数据使用期间保持存活"""

    @property
    def tensor(self) -> torch.Tensor:
        """返回 1D uint8 的零拷贝视图"""
        return self.pool_tensor[self.offset : self.offset + self.size]

    def as_tensor(self, dtype, shape) -> torch.Tensor:
        """返回带类型和形状的零拷贝视图"""
```

支持上下文管理器和析构器自动释放。

### 4. 连接器初始化

```python
class MooncakeTransferEngineConnector(OmniConnectorBase):
    supports_raw_data: bool = True  # 支持原始 bytes/Tensor 直传

    def __init__(self, config):
        # 1. Mooncake Engine 初始化
        self.engine = TransferEngine()
        self.engine.initialize(self.host, "P2PHANDSHAKE", self.protocol, self.device_name)

        # 2. 内存池分配与注册（支持 CPU pinned 和 GPU）
        self.pool = torch.empty(self.pool_size, dtype=torch.uint8).pin_memory()
        self.engine.register_memory(self.base_ptr, self.pool_size)
        self.allocator = BufferAllocator(self.pool_size)

        # 3. 角色决定（sender / receiver）
        # sender: 绑定 ZMQ 监听器
        # receiver: 跳过 ZMQ 绑定，仅接收
```

关键配置项：
- `role`: `"sender"` 或 `"receiver"`
- `protocol`: `"rdma"` 或 `"tcp"`
- `memory_pool_size`: 默认 1GB
- `memory_pool_device`: `"cpu"` 或 GPU 设备

### 5. put() — 生产者侧

```python
def put(self, from_stage, to_stage, put_key, data):
    # 根据数据类型选择路径：
    # - ManagedBuffer（同一内存池）: 零拷贝
    # - ManagedBuffer（不同内存池）: 回退到 tensor 拷贝
    # - torch.Tensor / bytes: 拷贝到内存池
    # - 其他（dict 等）: 先序列化再拷贝

    # 返回元数据（供 get() 使用）
    metadata = {
        "rdma_host": self.host,
        "rdma_zmq_port": self.zmq_port,
        "rdma_rpc_port": self.rpc_port,
        "data_size": size,
        "is_fast_path": is_fast_path,
    }
```

### 6. get() — 消费者侧

两种模式：
- **有元数据**（从 Orchestrator 通知中获得）：直接从元数据中获取 sender 地址
- **无元数据**（KV 缓存传输路径）：使用预配置的 `sender_host/sender_zmq_port` 查询

流程：
1. 在本地内存池分配接收缓冲区
2. 通过 ZMQ 与 sender 握手（发送目标地址和长度）
3. 等待 sender 通过 RDMA 写入数据
4. 收到 `TRANS_DONE` 后返回数据

### 7. ZMQ 监听器线程

```python
def _zmq_listener_loop(self):
    """Sender 侧的后台线程，处理 receiver 的拉取请求"""
    # 支持两种请求类型：
    # 1. QUERY_INFO: receiver 查询某个 key 的元数据
    # 2. MooncakeAgentMetadata: receiver 提供目标地址，sender 执行 RDMA 写入
```

### 8. 过期缓冲区清理

```python
_BUFFER_TTL_SECONDS = 300  # 5 分钟

def _purge_stale_buffers(self):
    """自动回收超过 TTL 的缓冲区，防止 receiver 崩溃导致内存泄漏"""
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `MooncakeTransferEngineConnector` | class | RDMA/TCP 高性能连接器 |
| `BufferAllocator` | class | 内存池分配器（空闲列表算法） |
| `ManagedBuffer` | class | 内存池中的托管缓冲区视图 |
| `MooncakeAgentMetadata` | dataclass | ZMQ 握手元数据 |
| `QueryRequest` / `QueryResponse` | dataclass | ZMQ 查询请求/响应 |
| `put()` | method | 生产者：暴露数据供 RDMA 传输 |
| `get()` | method | 消费者：通过 RDMA 拉取数据 |
| `update_sender_info()` | method | 动态注入 sender 地址（用于无元数据模式） |

## 与其他模块的关系

- 继承 `OmniConnectorBase`
- 使用 `OmniSerializer` 处理非原始类型数据
- 通过 `OmniConnectorFactory` 注册为 `"MooncakeTransferEngineConnector"`
- 依赖 `mooncake.engine.TransferEngine`、`zmq`、`msgspec`

## 总结

`MooncakeTransferEngineConnector` 是性能最高的连接器实现，通过 RDMA 零拷贝实现低延迟大数据传输。它包含完整的内存池管理（分配器 + 托管缓冲区）、ZMQ 握手协议、后台监听线程、过期清理机制等。当前限制为点对点通信（1:1），未来可扩展支持广播。
