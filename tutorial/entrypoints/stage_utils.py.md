# `stage_utils.py` — 阶段管理工具

## 文件概述

该文件提供了与多阶段管线相关的底层工具函数，包括动态函数加载、阶段任务类型定义、GPU 设备分配、共享内存 IPC 通信、JSONL 日志写入等功能。这些工具被编排器和阶段工作器广泛使用。

## 关键代码解析

### 动态函数加载

```python
def load_func_from_config(func_path: str | None) -> Callable[..., Any] | None:
    """从完全限定点分路径动态导入可调用对象"""
    module_path, func_name = func_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
```

用于从 YAML 配置中加载模型特定的处理函数（如 CFG 提示扩展函数）。

### 阶段任务类型

```python
class OmniStageTaskType(enum.Enum):
    GENERATE = "generate"
    ABORT = "abort"
    SHUTDOWN = "shutdown"
    PROFILER_START = "profiler_start"
    PROFILER_STOP = "profiler_stop"
    COLLECTIVE_RPC = "collective_rpc"
```

定义了阶段引擎支持的所有任务类型，用于编排器与阶段工作器之间的消息通信。

### GPU 设备分配

```python
def set_stage_devices(stage_id, devices, device_type=None):
    """为每个阶段配置 GPU 可见性"""
    # 支持逗号分隔的多设备: "2,5,7"
    if isinstance(devices, str) and "," in devices:
        # 将逻辑索引映射到物理设备
        mapping = [int(x) for x in vis.split(",")]
        mapped_devices = [str(mapping[idx]) for idx in toks]
        os.environ[env_var] = ",".join(mapped_devices)
```

关键特性：
- 支持 CUDA 和 NPU（华为昇腾）设备
- 逻辑索引到物理设备 ID 的映射
- 自动设置 `CUDA_VISIBLE_DEVICES` 或 `ASCEND_RT_VISIBLE_DEVICES`

### 共享内存 IPC

```python
def shm_write_bytes(payload: bytes, name=None) -> dict:
    """将字节写入共享内存，返回元数据 {name, size}"""
    shm = _shm.SharedMemory(create=True, size=len(payload), name=name)
    mv = memoryview(shm.buf)
    mv[:len(payload)] = payload
    return {"name": shm.name, "size": len(payload)}

def shm_read_bytes(meta: dict) -> bytes:
    """根据元数据从共享内存读取字节并清理"""
    shm = _shm.SharedMemory(name=meta["name"])
    data = bytes(memoryview(shm.buf)[:meta["size"]])
    shm.close(); shm.unlink()
    return data
```

用于跨进程传输大型数据（如模型输出），当序列化大小超过阈值时自动切换到共享内存通道。

### 自适应 IPC 编码

```python
def encode_for_ipc(obj, threshold, obj_key, shm_key) -> dict:
    """根据数据大小自动选择内联或共享内存传输"""
    payload = serialize_obj(obj)
    if len(payload) > threshold:
        return {shm_key: shm_write_bytes(payload)}
    else:
        return {obj_key: obj}
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `load_func_from_config()` | 函数 | 从配置路径动态加载函数 |
| `OmniStageTaskType` | 枚举 | 阶段任务类型定义 |
| `SHUTDOWN_TASK` | 常量 | 关闭任务消息 |
| `set_stage_devices()` | 函数 | 配置阶段 GPU 设备可见性 |
| `serialize_obj()` | 函数 | 序列化对象为字节 |
| `shm_write_bytes()` / `shm_read_bytes()` | 函数 | 共享内存读写 |
| `maybe_dump_to_shm()` | 函数 | 按阈值决定是否使用共享内存 |
| `encode_for_ipc()` | 函数 | 自适应 IPC 编码 |
| `append_jsonl()` | 函数 | 追加写入 JSONL 日志 |

## 与其他模块的关系

- 被编排器和阶段工作器广泛使用
- `OmniStageTaskType` 在阶段间消息通信中定义任务类型
- `set_stage_devices()` 在阶段工作器启动前调用
- 共享内存 IPC 函数被 `utils.py` 中的高层函数使用

## 总结

`stage_utils.py` 是多阶段管线的基础设施层，提供了设备管理、进程间通信、数据序列化和任务类型定义等底层能力。其自适应的 IPC 机制（内联 vs 共享内存）在保证小数据低延迟的同时支持大数据的高效传输。
