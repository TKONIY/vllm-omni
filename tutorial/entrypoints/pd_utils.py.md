# `pd_utils.py` — Prefill-Decode 分离工具

## 文件概述

该文件实现了 `PDDisaggregationMixin`，为多阶段管线提供 Prefill-Decode (PD) 分离支持。PD 分离是一种将 LLM 推理拆分为预填充（prefill）和解码（decode）两个独立阶段的优化技术，允许它们在不同的 GPU 上并行执行，通过 KV 缓存传输实现协同工作。

## 关键代码解析

### PD 对检测

```python
def _detect_pd_separation(self) -> tuple[int, int] | None:
    """扫描阶段列表，查找 prefill/decode 对"""
    for i, stage in enumerate(self.stage_list):
        if getattr(stage, "is_prefill_only", False):
            prefill_by_id[i] = i
        if getattr(stage, "is_decode_only", False):
            decode_indices.append(i)

    for j in decode_indices:
        source_ids = getattr(self.stage_list[j], "engine_input_source", [])
        for src in source_ids:
            if src in prefill_by_id:
                pd_pairs.append((prefill_by_id[src], j))
    return pd_pairs[0] if pd_pairs else None
```

通过检查阶段的 `is_prefill_only` 和 `is_decode_only` 标记，自动发现 PD 分离对。目前仅支持单对 PD 分离。

### 配置验证

```python
def _validate_pd_separation_config(self):
    # 验证 kv_role 配置正确
    if p_role not in ("kv_producer", "kv_both"):
        raise ValueError(...)
    if d_role not in ("kv_consumer", "kv_both"):
        raise ValueError(...)
    # 验证 connector 匹配
    if p_conn != d_conn:
        raise ValueError("PD connector mismatch")
    # 验证 tensor_parallel_size 一致
    if p_tp != d_tp:
        raise ValueError("PD stages must have matching tensor_parallel_size")
```

严格验证 PD 两端的配置一致性，包括 KV 角色、连接器类型、缓冲设备和张量并行度。

### 采样参数准备

```python
def _prepare_prefill_sampling_params(self, req_id, sp):
    sp = sp.clone()
    sp.max_tokens = 1  # Prefill 阶段只需生成 1 个 token
    sp.stop = []
    sp.extra_args["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": f"xfer-{req_id}",
    }
    return sp

def _build_decode_kv_params(self, req_id, sp_next, prefill_kv_params=None):
    decode_kv_params = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "transfer_id": f"xfer-{req_id}",
    }
    # 注入 Mooncake 引导地址等连接器信息
    if self._pd_connector_info:
        baddr = self._pd_connector_info.get("prefill_bootstrap_addr")
        if baddr is not None:
            decode_kv_params["remote_bootstrap_addr"] = baddr
```

Prefill 阶段设置 `max_tokens=1` 并标记为 KV 生产者；Decode 阶段标记为 KV 消费者，注入远程引导地址。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `PDDisaggregationMixin` | Mixin 类 | PD 分离的所有辅助方法 |
| `_detect_pd_separation()` | 方法 | 自动检测 PD 分离对 |
| `_validate_pd_separation_config()` | 方法 | 验证 PD 配置一致性 |
| `_prepare_prefill_sampling_params()` | 方法 | 构建 Prefill 阶段采样参数 |
| `_build_decode_kv_params()` | 方法 | 构建 Decode 阶段 KV 传输参数 |
| `_prepare_pd_decode_routing()` | 方法 | 准备 Prefill->Decode 路由 |
| `_is_pd_routing()` | 方法 | 判断是否为 PD 边界 |
| `_maybe_expand_sampling_params()` | 方法 | 自动补充 Decode 阶段采样参数 |

## 与其他模块的关系

- 作为 Mixin 注入 `OmniBase` 的子类中
- 依赖 `OmniStage` 的阶段属性（`is_prefill_only`、`engine_input_source` 等）
- 与 Mooncake 等 KV 传输连接器配合使用
- 通过 `kv_transfer_params` 与 vLLM 的 KV 缓存传输系统对接

## 总结

`PDDisaggregationMixin` 实现了完整的 Prefill-Decode 分离支持，从自动检测 PD 对、配置验证到采样参数构建和路由准备。该机制允许将计算密集的预填充和内存密集的解码部署在不同设备上，通过 KV 缓存传输实现高效协作。
