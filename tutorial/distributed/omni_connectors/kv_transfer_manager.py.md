# `kv_transfer_manager.py` — KV 缓存传输管理器

## 文件概述

该文件实现了 `OmniKVTransferManager`，统一管理 OmniConnector 和 KV 缓存传输的全部生命周期：连接器初始化、KV 缓存从 GPU 提取、通过连接器传输、接收并还原到目标设备。主要用于多阶段推理中的 KV 缓存跨阶段转移场景。

## 关键代码解析

### 1. 配置数据类

```python
@dataclass
class OmniKVCacheConfig:
    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0
```

```python
@dataclass
class KVCacheTransferData:
    request_id: str
    layer_blocks: dict[str, Any]  # {"key_cache": [...], "value_cache": [...]}
    block_ids: list[int]
    metadata: dict[str, Any]      # block_size, num_layers, dtype, seq_len
```

### 2. 多种创建方式

管理器支持从不同来源创建，适应不同的运行场景：

```python
@classmethod
def from_model_config(cls, config):     # AR 模型运行器
@classmethod
def from_od_config(cls, config):        # Diffusion 运行器
@classmethod
def from_vllm_config(cls, vllm_config, model_config):  # vLLM 配置 + 回退到 kv_transfer_config
```

### 3. 连接器惰性初始化

```python
@property
def connector(self):
    if self._connector is False:  # 之前初始化失败的哨兵值
        return None
    if self._connector is None:
        cfg = self.config.connector_config
        if cfg and (c_type := cfg.get("type")):
            try:
                self._connector = OmniConnectorFactory.create_connector(
                    ConnectorSpec(name=c_type, extra=...)
                )
            except Exception:
                self._connector = False  # 缓存失败，避免热路径重复尝试
    return self._connector if self._connector else None
```

使用 `False` 作为失败哨兵值，避免在推理热路径中反复尝试初始化。

### 4. KV 缓存提取流程

```python
def _extract_kv_cache(self, req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype, ...):
```

逐层处理 KV 缓存：
1. 调用 `normalize_layer_kv()` 统一不同注意力后端的 KV 布局
2. 验证 block ID 有效性
3. 按 block ID 选取 key/value 块，展平并截断到 `seq_len`
4. 从 GPU 拷贝到 CPU（`detach().cpu().contiguous()`）

### 5. 带重试的传输

```python
def _transfer_with_retry(self, from_stage, to_stage, request_id, data, max_retries=3):
    for attempt in range(max_retries):
        success, size, metadata = self.connector.put(...)
        if success:
            return success, size, metadata
        time.sleep(0.1 * (2 ** attempt))  # 指数退避
```

### 6. KV 缓存接收与应用

```python
def receive_kv_cache_for_request(self, request_id, target_device=None):
    while True:
        result = self.connector.get(...)
        if result:
            # 将 tensor 移动到目标设备
            for tensor in cache_list:
                cache_list[i] = tensor.to(target_device).contiguous()
            return data, size
        if time.time() - start_time > timeout:
            return None, 0
        time.sleep(0.5)

def apply_kv_cache_to_request(self, req, data):
    kv_obj = SimpleNamespace(**layer_blocks)
    req.past_key_values = kv_obj
    req.sampling_params.past_key_values = kv_obj  # BagelPipeline 兼容
```

### 7. CFG KV 缓存支持

```python
def receive_multi_kv_cache(self, req, cfg_kv_collect_func=None, target_device=None):
```

除接收主 KV 缓存外，还支持接收 CFG（Classifier-Free Guidance）伴随 KV 缓存。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniKVCacheConfig` | dataclass | KV 缓存传输配置 |
| `KVCacheTransferData` | dataclass | KV 缓存传输数据容器 |
| `OmniKVTransferManager` | class | KV 缓存传输管理器 |
| `handle_finished_requests_kv_transfer()` | method | 批量处理已完成请求的 KV 传输 |
| `receive_kv_cache_for_request()` | method | 接收指定请求的 KV 缓存 |
| `apply_kv_cache_to_request()` | method | 将 KV 缓存应用到请求对象 |
| `receive_multi_kv_cache()` | method | 接收主 KV + CFG 伴随 KV |

## 与其他模块的关系

- 使用 `OmniConnectorFactory` 创建连接器
- 使用 `normalize_layer_kv()` 处理不同注意力后端的 KV 布局
- 被 AR 模型运行器和 Diffusion 模型运行器使用
- 支持 BagelPipeline 的 `past_key_values` 接口

## 总结

`OmniKVTransferManager` 是 KV 缓存跨阶段传输的核心管理类。它封装了从 GPU 提取 KV 缓存、序列化传输、接收还原到目标设备的完整流程，支持惰性初始化、指数退避重试、超时控制和 CFG 伴随 KV 缓存等高级特性。
