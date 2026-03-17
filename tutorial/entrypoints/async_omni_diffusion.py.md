# `async_omni_diffusion.py` — 异步扩散模型推理入口

## 文件概述

`AsyncOmniDiffusion` 为扩散模型（如图像生成、视频生成）提供异步推理接口。与 `AsyncOmni` 的多阶段 LLM 管线不同，它直接包装 `DiffusionEngine`，通过线程池将同步的扩散推理转为异步操作。

## 关键代码解析

### 初始化与模型配置

```python
class AsyncOmniDiffusion:
    def __init__(self, model: str, od_config=None, **kwargs):
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(model=model, **kwargs)

        # 自动检测模型类型：尝试 model_index.json，回退到 config.json
        try:
            config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
            od_config.model_class_name = config_dict.get("_class_name", None)
        except ...:
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            model_type = cfg.get("model_type")
            # 特殊处理 Bagel、NextStep 等非标准模型
```

初始化时自动探测模型类型（diffusers 标准模型读 `model_index.json`，非标准模型读 `config.json`），并对 Bagel、NextStep 等自定义架构做特殊处理。

### 异步生成

```python
async def generate(self, prompt, sampling_params, request_id=None, lora_request=None):
    request = OmniDiffusionRequest(prompts=[prompt], ...)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(self._executor, self.engine.step, request)
    return result[0]
```

核心设计：使用单线程 `ThreadPoolExecutor` 将同步的 `DiffusionEngine.step()` 调用包装为异步操作。每次提交单个请求。

### 资源管理

```python
self._weak_finalizer = weakref.finalize(
    self, _weak_close_async_omni_diffusion, self.engine, self._executor,
)
```

通过 `weakref.finalize` 实现 GC 安全的资源清理，即使用户忘记调用 `close()` 也能释放引擎和线程池。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `AsyncOmniDiffusion` | 类 | 异步扩散模型推理入口 |
| `generate()` | 异步方法 | 异步图像/视频生成 |
| `generate_stream()` | 异步生成器 | 流式生成（当前为单次输出） |
| `close()` / `shutdown()` | 方法 | 释放引擎和线程池资源 |
| `add_lora()` / `remove_lora()` | 异步方法 | LoRA 适配器管理 |
| `start_profile()` / `stop_profile()` | 异步方法 | 性能分析控制 |
| `_weak_close_async_omni_diffusion()` | 函数 | GC 析构时的清理回调 |

## 与其他模块的关系

- 底层使用 `DiffusionEngine`（`vllm_omni.diffusion`）执行实际推理
- 使用 `OmniDiffusionConfig` 和 `OmniDiffusionRequest` 数据结构
- 被 `openai/api_server.py` 用于独立的扩散模型服务
- 与 `AsyncOmni` 互补：`AsyncOmni` 处理 LLM 管线，`AsyncOmniDiffusion` 处理纯扩散模型

## 总结

`AsyncOmniDiffusion` 是扩散模型推理的异步封装层，采用线程池桥接模式将同步推理转为异步。它自动检测模型类型，支持 LoRA 热加载和性能分析，并通过弱引用确保资源安全释放。
