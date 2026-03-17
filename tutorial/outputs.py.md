# `outputs.py` — 输出数据结构

## 文件概述

`outputs.py` 定义了 vllm-omni 的两个核心输出类：

1. **`OmniModelRunnerOutput`**：模型执行层的输出，扩展 vLLM 的 `ModelRunnerOutput`，增加多模态输出字段
2. **`OmniRequestOutput`**：面向用户的统一请求输出，同时支持流水线模式和扩散模型模式

## 关键代码解析

### OmniModelRunnerOutput

```python
class OmniModelRunnerOutput(ModelRunnerOutput):
    multimodal_outputs: dict[str, torch.Tensor] | None = None
    kv_extracted_req_ids: list[str] | None = None
```

- `multimodal_outputs`：按模态名称映射的输出张量，如 `{"image": tensor, "audio": tensor}`
- `kv_extracted_req_ids`：KV 缓存已从 GPU 提取到 CPU 的请求 ID 列表，调度器据此释放 block table

### OmniRequestOutput — 双模式工厂方法

```python
@classmethod
def from_pipeline(cls, stage_id, final_output_type, request_output):
    """从流水线阶段创建输出"""
    return cls(
        request_id=getattr(request_output, "request_id", ""),
        stage_id=stage_id,
        final_output_type=final_output_type,
        request_output=request_output,
        finished=True,
    )

@classmethod
def from_diffusion(cls, request_id, images, prompt=None, ...):
    """从扩散模型创建输出"""
    return cls(
        request_id=request_id,
        final_output_type=final_output_type,
        images=images,
        ...
    )
```

通过工厂方法清晰区分两种输出来源，避免构造函数参数混乱。

### vLLM 兼容性透传属性

```python
@property
def prompt_token_ids(self) -> list[int] | None:
    if self.request_output is not None:
        return getattr(self.request_output, "prompt_token_ids", None)
    return None

@property
def outputs(self) -> list[Any]:
    if self.request_output is not None:
        return getattr(self.request_output, "outputs", [])
    return []
```

这些属性将 vLLM 服务层所需的字段透传到底层的 `RequestOutput`，确保 `OmniRequestOutput` 可以无缝替代 `RequestOutput` 使用。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniModelRunnerOutput` | 类 | 模型执行器输出，携带多模态张量 |
| `OmniRequestOutput` | 数据类 | 统一的用户请求输出 |
| `from_pipeline` | 类方法 | 从流水线阶段创建输出 |
| `from_diffusion` | 类方法 | 从扩散模型创建输出 |
| `multimodal_output` | 属性 | 获取多模态输出字典 |
| `custom_output` | 属性 | 获取自定义输出数据 |
| `is_diffusion_output` | 属性 | 判断是否为扩散模型输出 |
| `is_pipeline_output` | 属性 | 判断是否为流水线输出 |
| `to_dict` | 方法 | 转换为可 JSON 序列化的字典 |

## 与其他模块的关系

- 继承 `vllm.v1.outputs.ModelRunnerOutput` 和 `vllm.outputs.RequestOutput`
- 被调度器 (`core/sched/`) 用于包装模型执行结果
- 被入口点 (`entrypoints/`) 用于返回给用户
- 引用 `inputs.data.OmniPromptType` 用于记录原始 prompt

## 总结

`outputs.py` 是 vllm-omni 输出体系的核心，通过统一的 `OmniRequestOutput` 类封装了流水线和扩散模型两种截然不同的输出模式，并通过透传属性保持了与 vLLM 服务层的完全兼容。
