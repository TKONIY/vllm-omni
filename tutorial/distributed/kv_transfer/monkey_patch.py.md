# `monkey_patch.py` — Mooncake 连接器 PD 分离补丁

## 文件概述

本文件实现了对 vLLM 原生 `MooncakeConnector` 的 monkey-patch，通过继承并重写关键方法来解决 Prefill-Decode 分离部署中 request-ID 不匹配的问题。补丁完成后，Prefill 端的 `remote_request_id` 会被注入到 `kv_transfer_params` 中传递给 Decode 端。

## 关键代码解析

### 1. PatchedRecvReqMeta 数据类

```python
@dataclass
class PatchedRecvReqMeta:
    """Receive-request metadata carrying the prefill engine's request ID."""
    request_id: str
    remote_request_id: str
    local_block_ids: list[int]
    kv_transfer_params: dict[str, Any]
```

携带了 Prefill 端真正的 request ID（`remote_request_id`），以便 Decode 端用它作为 ZMQ 查找键。

### 2. request_finished 方法补丁

```python
def request_finished(self, request, block_ids):
    result = super().request_finished(request, block_ids)
    # ...
    if kv_params is not None and isinstance(kv_params, dict):
        kv_params["remote_request_id"] = req_id or "NOT_SET"
```

当 Prefill 请求完成时，将当前请求的 `request_id` 注入到返回的 `kv_transfer_params` 中。这样 Decode 端就知道要用哪个 ID 去查找 KV 数据。

### 3. add_new_req 方法补丁

```python
def add_new_req(self, request_id, local_block_ids, kv_transfer_params=None, **kwargs):
    super().add_new_req(request_id, local_block_ids, kv_transfer_params, **kwargs)
    if load_remote_cache:
        remote_request_id = kv_transfer_params.get("remote_request_id", request_id)
        meta = PatchedRecvReqMeta(
            request_id=request_id,
            remote_request_id=remote_request_id,
            local_block_ids=local_block_ids,
            kv_transfer_params=kv_transfer_params,
        )
        self._reqs_need_recv[request_id] = meta
```

Decode 端添加新请求时，从 `kv_transfer_params` 中提取 `remote_request_id`，构建补丁版元数据。

### 4. group_kv_pull 方法补丁（核心）

```python
def group_kv_pull(self, metadata=None):
    """Use remote_request_id as ZMQ lookup key via save-patch-restore."""
    original_recv = self._reqs_need_recv.copy()
    patched_recv = {}
    for local_id, meta in original_recv.items():
        if isinstance(meta, PatchedRecvReqMeta):
            remote_id = meta.remote_request_id
            self.remote_to_local_req[remote_id] = local_id
            patched_meta = type(meta)(
                request_id=remote_id,  # 用 remote_id 替换
                ...
            )
            patched_recv[remote_id] = patched_meta
    self._reqs_need_recv = patched_recv
    super().group_kv_pull(metadata)
    # 恢复未消费的条目
```

关键逻辑：临时将 `_reqs_need_recv` 中的 key 从本地 ID 替换为远程 ID，调用原始 `group_kv_pull` 执行 ZMQ 查找，然后恢复未消费的条目。

### 5. apply_mooncake_connector_patch 函数

```python
def apply_mooncake_connector_patch() -> bool:
    _mc_module.MooncakeConnector = PatchedClass
    for _, module in sys.modules.items():
        if hasattr(module, "MooncakeConnector") and module.MooncakeConnector is _OriginalClass:
            module.MooncakeConnector = PatchedClass
```

全局替换：在 Mooncake 模块和所有已导入该类的模块中，将原始类替换为补丁类。使用全局标志 `_patched` 确保只执行一次。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `PatchedRecvReqMeta` | dataclass | 携带 `remote_request_id` 的接收请求元数据 |
| `PatchedMooncakeConnector` | class | 修补后的 MooncakeConnector 子类 |
| `apply_mooncake_connector_patch()` | function | 应用全局 monkey-patch |
| `_import_mooncake_module()` | function | 兼容 vLLM >=0.16 和旧版本的导入逻辑 |
| `_create_patched_mooncake_connector()` | function | 创建补丁子类（延迟导入） |

## 与其他模块的关系

- 依赖 vLLM 的 `MooncakeConnector`（通过动态导入）
- 被上层启动逻辑（如 engine 初始化时）调用 `apply_mooncake_connector_patch()` 来激活
- 与 `omni_connectors` 模块独立——此补丁作用于 vLLM 原生的 KV transfer 路径，而非 omni 自有的传输通道

## 总结

`monkey_patch.py` 通过继承和方法重写解决了 PD 分离场景下 vLLM request-ID 不一致的核心问题。补丁采用"保存-替换-恢复"策略将远程 ID 临时注入查找流程，实现了对原始代码的最小侵入。
