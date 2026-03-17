# `omni.py` — 同步推理入口

## 文件概述

`Omni` 类是面向离线批量推理的同步入口，继承自 `OmniBase`。它适用于 Python 脚本中的批量处理场景，支持多提示并发、进度条显示和生成器模式，是 `AsyncOmni` 的同步对应物。

## 关键代码解析

### 同步生成方法

```python
class Omni(OmniBase):
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list=None,
        *,
        py_generator: bool = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        sampling_params_list = self.resolve_sampling_params_list(sampling_params_list)
        if py_generator:
            return self._run_generation_with_generator(prompts, sampling_params_list, use_tqdm)
        return list(self._run_generation(prompts, sampling_params_list, use_tqdm))
```

通过 `py_generator` 参数控制返回模式：
- `False`（默认）: 等待所有请求完成后返回列表
- `True`: 返回 Python 生成器，边生成边产出结果

### 核心生成循环

```python
def _run_generation(self, prompts, sampling_params_list, use_tqdm):
    # 1. 强制 LLM 阶段使用 FINAL_ONLY 输出模式
    sampling_params_list = self._set_final_only_for_llm_stages(sampling_params_list)

    # 2. 为每个提示创建请求并提交
    for req_id, prompt in zip(request_ids, request_prompts):
        self.engine.add_request(request_id=req_id, prompt=prompt, ...)

    # 3. 轮询输出直到所有请求完成
    while active_reqs:
        msg = self.engine.try_get_output()
        should_continue, req_id, stage_id, req_state = self._handle_output_message(msg)
        if should_continue:
            continue
        output_to_yield = self._process_single_result(result=msg, ...)
        if output_to_yield is not None:
            yield output_to_yield
        if msg.get("finished"):
            active_reqs.discard(req_id)
```

关键设计：
1. LLM 阶段强制 `FINAL_ONLY` 模式，避免产生中间 token 输出
2. 所有提示一次性提交，然后通过轮询收集结果
3. 使用 tqdm 进度条显示完成进度
4. 异常时自动中止所有未完成的请求

### 采样参数优化

```python
def _set_final_only_for_llm_stages(self, sampling_params_list):
    for stage_id, params in enumerate(sampling_params_list):
        sp = copy.deepcopy(params)
        stage_meta = self.engine.get_stage_metadata(stage_id)
        if stage_meta.get("stage_type") != "diffusion" and hasattr(sp, "output_kind"):
            sp.output_kind = RequestOutputKind.FINAL_ONLY
        effective_params.append(sp)
    return effective_params
```

离线模式不需要流式输出，因此将所有非扩散阶段的输出设为 `FINAL_ONLY`，减少不必要的中间输出传输。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `Omni` | 类 | 同步离线推理入口 |
| `generate()` | 方法 | 同步生成（支持列表和生成器两种返回模式） |
| `_run_generation()` | 生成器 | 核心生成循环 |
| `_set_final_only_for_llm_stages()` | 方法 | 优化 LLM 阶段为仅最终输出 |
| `abort()` | 方法 | 同步中止请求 |

## 与其他模块的关系

- 继承 `OmniBase`（`omni_base.py`），复用引擎管理和消息处理逻辑
- 使用 `ClientRequestState`（`client_request_state.py`）跟踪请求
- 底层通过 `AsyncOmniEngine` 的同步接口（`add_request`、`try_get_output`）工作
- 在 `__init__.py` 中导出，可通过 `from vllm_omni.entrypoints import Omni` 使用

## 总结

`Omni` 为离线批量推理提供了简洁的同步接口，支持多提示并发处理和进度显示。通过将 LLM 阶段强制为 `FINAL_ONLY` 模式来优化离线场景的性能。它与 `AsyncOmni` 共享相同的底层引擎和消息处理逻辑，区别仅在于执行模式。
