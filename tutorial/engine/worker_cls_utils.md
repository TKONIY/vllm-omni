# `worker_cls_utils.py` — Worker 类解析工具

## 文件概述

`worker_cls_utils.py` 是一个简洁的工具模块，仅包含一个函数 `resolve_worker_cls()`。它根据引擎参数中的 `worker_type` 字段，自动解析并设置对应的 Worker 类路径。vLLM-Omni 支持两种 Worker 类型：`ar`（自回归）和 `generation`（生成），分别对应不同的模型执行逻辑。

## 关键代码解析

### resolve_worker_cls — Worker 类解析

```python
def resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    worker_type = engine_args.get("worker_type", None)
    if not worker_type:
        return  # 未指定 worker_type，使用 vLLM 默认 Worker

    worker_cls = engine_args.get("worker_cls")
    if worker_cls is not None and worker_cls != "auto":
        return  # 用户已显式指定 worker_cls，不覆盖

    from vllm_omni.platforms import current_omni_platform

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")
```

解析逻辑：

1. **检查 worker_type**：如果未指定（None 或空字符串），直接返回，使用 vLLM 默认 Worker
2. **尊重显式配置**：如果用户已经指定了 `worker_cls` 且不是 `"auto"`，不进行覆盖
3. **平台适配**：通过 `current_omni_platform` 获取平台特定的 Worker 类路径
   - `"ar"` -> `get_omni_ar_worker_cls()`：自回归 Worker，用于流式生成（如 TTS 的 AR 阶段）
   - `"generation"` -> `get_omni_generation_worker_cls()`：生成 Worker，用于非自回归生成（如 flow matching）
4. **错误处理**：对未知的 `worker_type` 抛出 `ValueError`

该函数直接修改传入的字典（就地修改），设置 `worker_cls` 键的值。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `resolve_worker_cls()` | 函数 | 根据 worker_type 解析并设置 Worker 类路径 |

## 与其他模块的关系

- **`stage_init.py`**：`build_engine_args_dict()` 中调用 `resolve_worker_cls()` 为每个 LLM 阶段设置 Worker 类
- **`vllm_omni/platforms/`**：`current_omni_platform` 提供平台特定的 Worker 类路径（CUDA/其他加速器）
- **vLLM Worker 系统**：解析后的 `worker_cls` 字符串被 vLLM 用于动态加载 Worker 类

## 总结

`worker_cls_utils.py` 虽然只有一个函数，但它是多阶段流水线中 Worker 类型路由的关键环节。通过将 `worker_type`（语义化的类型名）映射为 `worker_cls`（具体的类路径），它实现了平台无关的 Worker 类型选择。`"ar"` 和 `"generation"` 两种类型分别对应自回归和非自回归的推理模式，这是 TTS 等多阶段模型的基本架构划分。
