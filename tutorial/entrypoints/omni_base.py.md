# `omni_base.py` — 共享运行时基础

## 文件概述

`OmniBase` 是 `AsyncOmni` 和 `Omni` 的共享基类，封装了引擎初始化、模型下载、请求状态管理、输出消息处理和性能指标收集等共用逻辑。该文件还包含模型快照下载和 GC 安全清理的辅助函数。

## 关键代码解析

### 模型下载

```python
def omni_snapshot_download(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download
        return snapshot_download(model_id)
    download_weights_from_hf_specific(model_name_or_path=model_id, ...)
    return model_id
```

支持从 HuggingFace Hub 和 ModelScope 下载模型权重，优先使用本地路径。

### 引擎初始化

```python
class OmniBase:
    def __init__(self, model: str, **kwargs):
        model = omni_snapshot_download(model)
        self.engine = AsyncOmniEngine(
            model=model, engine_args=engine_args,
            init_timeout=init_timeout, stage_init_timeout=stage_init_timeout,
            **kwargs,
        )
        self._weak_finalizer = weakref.finalize(self, _weak_shutdown_engine, self.engine)
        self.request_states: dict[str, ClientRequestState] = {}
        self.default_sampling_params_list = self.engine.default_sampling_params_list
```

初始化 `AsyncOmniEngine` 作为底层编排器，并通过 `weakref.finalize` 确保引擎在 GC 时安全关闭。`request_states` 字典管理所有活跃请求的状态。

### 输出消息处理

```python
def _handle_output_message(self, msg) -> OutputMessageHandleResult:
    """处理编排器输出队列的单条消息"""
    msg_type = msg.get("type")
    if msg_type == "stage_metrics":
        self._process_stage_metrics_message(msg)
        return True, None, None, None  # 继续
    if msg_type == "error":
        raise RuntimeError(msg.get("error"))
    if msg_type != "output":
        return True, None, None, None  # 跳过
    # 查找请求状态并返回
    req_state = self.request_states.get(req_id)
    return False, req_id, stage_id, req_state
```

消息处理采用类型分发模式：`stage_metrics` 类型更新性能指标，`error` 类型抛出异常，`output` 类型返回给调用者处理。

### 结果构造

```python
def _process_single_result(self, result, stage_id, metrics, ...):
    stage_meta = self.engine.get_stage_metadata(stage_id)
    if not stage_meta["final_output"]:
        return None  # 中间阶段的输出不返回给用户
    return OmniRequestOutput(
        stage_id=stage_id,
        final_output_type=stage_meta["final_output_type"],
        request_output=engine_outputs,
        images=images,
    )
```

只有标记为 `final_output` 的阶段输出才会返回给用户，中间阶段的输出被过滤。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniBase` | 类 | AsyncOmni 和 Omni 的共享基类 |
| `omni_snapshot_download()` | 函数 | 下载模型快照 |
| `_weak_shutdown_engine()` | 函数 | GC 时的引擎清理回调 |
| `resolve_sampling_params_list()` | 方法 | 标准化采样参数列表 |
| `_handle_output_message()` | 方法 | 处理编排器输出消息 |
| `_process_single_result()` | 方法 | 将原始结果构造为 OmniRequestOutput |
| `_compute_final_stage_id()` | 方法 | 根据输出模态计算最终阶段 ID |
| `_log_summary_and_cleanup()` | 方法 | 记录性能摘要并清理请求状态 |
| `shutdown()` | 方法 | 关闭引擎 |

## 与其他模块的关系

- 被 `AsyncOmni` 和 `Omni` 继承
- 管理 `AsyncOmniEngine`（`vllm_omni.engine`）实例
- 使用 `ClientRequestState` 跟踪请求
- 使用 `OrchestratorAggregator`（`vllm_omni.metrics`）收集性能指标
- 通过 `utils.py` 中的 `get_final_stage_id_for_e2e` 计算端到端指标的最终阶段

## 总结

`OmniBase` 是整个入口体系的基石，将引擎生命周期管理、请求状态追踪、消息路由和性能指标收集等横切关注点统一封装，使 `AsyncOmni` 和 `Omni` 可以专注于各自的执行模式（异步/同步）。
