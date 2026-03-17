# `bagel.py` -- Bagel CFG 提示扩展与 KV 缓存收集

## 文件概述

`bagel.py` 实现了 Bagel 图像生成模型的 Classifier-Free Guidance (CFG) 相关处理逻辑。Bagel 的 3 分支 CFG 要求在 AR 阶段同时处理多个提示变体（条件/无条件），然后在扩散阶段收集对应的 KV 缓存。本文件提供了两个核心函数：提示扩展（`expand_cfg_prompts`）和 KV 缓存收集（`collect_cfg_kv_caches`）。

**文件路径**: `/home/yangshen/code/vllm-omni/vllm_omni/model_executor/stage_input_processors/bagel.py`

## 关键代码解析

### CFG 三分支架构

Bagel 的 CFG 需要三个分支的推理结果：
- **gen（条件分支）**: 用户原始提示
- **cfg_text（文本无条件分支）**: 负面/空提示
- **cfg_img（图像无条件分支）**: 去除图像的提示

### expand_cfg_prompts -- 提示扩展

```python
def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
```

根据模态类型扩展提示：
- **text2img 模式**: 生成 1 个额外的 `cfg_text` 提示（负面提示）
- **img2img 模式**: 生成 2 个额外提示（`cfg_text` + `cfg_img`）
- **text2text / img2text**: 不扩展（返回空列表）

负面提示的获取优先级：
1. `prompt["negative_prompt"]`
2. `sampling_params.extra_args["negative_prompt"]`
3. 默认值 `"<|im_start|><|im_end|>"`

### collect_cfg_kv_caches -- KV 缓存收集

```python
def collect_cfg_kv_caches(
    request_id: str,
    cfg_request_ids: dict[str, str],
    kv_transfer_manager: Any,
    target_device: Any | None = None,
) -> dict[str, Any]:
```

在扩散阶段收集各 CFG 分支的 KV 缓存，通过 `kv_transfer_manager` 获取伴随请求的缓存数据。

### ExpandedPrompt 数据类

```python
@dataclass
class ExpandedPrompt:
    prompt: dict[str, Any] | str
    role: str                  # "cfg_text" 或 "cfg_img"
    request_id_suffix: str     # 如 "__cfg_text"
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `ExpandedPrompt` | dataclass | 扩展提示数据结构 |
| `expand_cfg_prompts` | 函数 | 将用户提示扩展为 CFG 多分支提示 |
| `collect_cfg_kv_caches` | 函数 | 收集各 CFG 分支的 KV 缓存 |
| `_get_negative_prompt` | 函数 | 解析负面提示 |
| `CFG_TEXT_SUFFIX` | 常量 | `"__cfg_text"` |
| `CFG_IMG_SUFFIX` | 常量 | `"__cfg_img"` |

## 与其他模块的关系

- **stage_configs/bagel.yaml**: 通过 `prompt_expand_func` 和 `cfg_kv_collect_func` 引用本文件中的函数
- **engine/**: 引擎在处理 Bagel 请求时调用 `expand_cfg_prompts` 生成伴随请求
- **models/bagel/**: Bagel 模型的 KV 缓存输出被此处的收集函数消费

## 总结

`bagel.py` 是 Bagel 模型 CFG 多分支推理的核心处理器，它将单个用户请求扩展为最多 3 个并行请求（gen + cfg_text + cfg_img），然后在扩散阶段收集各分支的 KV 缓存，实现 Classifier-Free Guidance 的多分支引导机制。
