# `cfg_companion_tracker.py` — CFG 伴随请求跟踪器

## 文件概述

该文件实现了 `CfgCompanionTracker`，用于管理 Classifier-Free Guidance (CFG) 中的伴随请求（companion requests）。在扩散模型中，CFG 需要同时运行有条件和无条件的推理路径。该跟踪器封装了提示扩展、父子 ID 映射、完成跟踪、延迟转发和超时处理等所有记账逻辑，保持编排器主循环的简洁。

## 关键代码解析

### 提示扩展

```python
def expand_prompts(self, request_id_to_prompt):
    pairs = []
    for rid, prompt in request_id_to_prompt.items():
        expanded = self._expand_func(prompt, self._sp0)
        for ep in expanded:
            cid = f"{rid}{ep.request_id_suffix}"
            role_map[ep.role] = cid
            self._companion_ids.add(cid)
            self._companion_to_parent[cid] = rid
            pairs.append((cid, ep.prompt))
    return pairs
```

调用模型特定的扩展函数，为每个用户请求创建 CFG 伴随请求（如无条件提示），并建立父子关系映射。

### 完成跟踪与延迟转发

```python
def on_companion_completed(self, companion_id):
    parent_id = self._companion_to_parent.get(companion_id)
    self._done[parent_id].add(companion_id)
    if parent_id in self._pending_parents and self.all_companions_done(parent_id):
        return parent_id  # 所有伴随请求完成，可以转发父请求
    return None

def defer_parent(self, parent_id, engine_outputs, stage_id):
    self._pending_parents[parent_id] = {
        "engine_outputs": engine_outputs,
        "stage_id": stage_id,
        "pending_since": time.time(),
    }
```

父请求先完成时会被挂起（defer），等待所有 CFG 伴随请求完成后再统一转发到下一阶段。

### 带 CFG KV 的转发

```python
def forward_parent_with_cfg(self, req_id, parent_result, ...):
    sp_next = copy.deepcopy(sampling_params_list[next_stage_id])
    if isinstance(sp_next, OmniDiffusionSamplingParams):
        sp_next.cfg_kv_request_ids = self.get_companion_request_ids(req_id)
    # 通过 connector 发送到下一阶段
```

转发时将 CFG 伴随请求的 ID 注入下一阶段的采样参数，使扩散阶段能够获取到对应的 KV 缓存。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `CfgCompanionTracker` | 类 | CFG 伴随请求的全生命周期管理 |
| `expand_prompts()` | 方法 | 将用户提示扩展为 CFG 伴随请求 |
| `on_companion_completed()` | 方法 | 标记伴随请求完成 |
| `on_companion_error()` | 方法 | 处理伴随请求失败 |
| `defer_parent()` | 方法 | 挂起等待伴随的父请求 |
| `check_timeouts()` | 方法 | 检查并清理超时的挂起请求 |
| `forward_parent_with_cfg()` | 方法 | 带 CFG KV 信息转发到下一阶段 |

## 与其他模块的关系

- 被编排器（Orchestrator）在生成循环中使用
- 使用 `OmniDiffusionSamplingParams` 注入 CFG KV 请求 ID
- 通过 `try_send_via_connector`（`distributed.omni_connectors`）实现跨阶段数据传输

## 总结

`CfgCompanionTracker` 将 CFG 机制的复杂记账逻辑从编排器主循环中解耦出来，提供了提示扩展、父子关系追踪、延迟转发和超时保护的完整方案，是多阶段扩散推理中不可或缺的协调组件。
