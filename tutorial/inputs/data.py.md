# `inputs/data.py` — 输入数据类型定义

## 文件概述

`data.py` 定义了 vllm-omni 的全套输入数据类型，扩展 vLLM 的 `TextPrompt`、`TokensPrompt`、`EmbedsPrompt` 和 `TokenInputs`，添加嵌入传递、附加信息和扩散模型采样参数等支持。

## 关键代码解析

### OmniTextPrompt — 扩展文本提示

```python
class OmniTextPrompt(TextPrompt):
    negative_prompt: NotRequired[str]
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[torch.Tensor]
    additional_information: NotRequired[dict[str, Any]]
```

在标准文本提示基础上增加：
- `negative_prompt`：反向提示词（扩散模型使用）
- `prompt_embeds`：预计算的提示嵌入
- `additional_information`：跨阶段传递的附加数据

### OmniTokensPrompt — 扩展 Token 提示

```python
class OmniTokensPrompt(TokensPrompt):
    negative_prompt: NotRequired[str]
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[list[torch.Tensor] | None]
    additional_information: NotRequired[dict[str, Any]]
```

### OmniTokenInputs — 扩展 Token 输入

```python
class OmniTokenInputs(TokenInputs):
    negative_prompt: NotRequired[str]
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[list[torch.Tensor] | None]
    additional_information: NotRequired[dict[str, Any]]
```

### OmniCustomPrompt — 自定义扩散提示

```python
class OmniCustomPrompt(TypedDict, total=False):
    prompt_ids: list[int] | list[list[int]]
    negative_prompt_ids: list[int] | list[list[int]]
    prompt_mask: torch.Tensor
    negative_prompt_mask: torch.Tensor
    extra_args: dict[str, Any]
```

允许直接传入预 tokenize 的输入，绕过 tokenization 阶段。

### token_inputs_omni — 构造辅助函数

```python
def token_inputs_omni(
    prompt_token_ids, prompt=None, cache_salt=None,
    prompt_embeds=None, additional_information=None,
) -> OmniTokenInputs:
    inputs = OmniTokenInputs(type="token", prompt_token_ids=prompt_token_ids)
    if prompt_embeds is not None:
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information is not None:
        inputs["additional_information"] = additional_information
    return inputs
```

### OmniDiffusionSamplingParams — 扩散采样参数

```python
@dataclass
class OmniDiffusionSamplingParams:
    # 基础参数
    num_inference_steps: int = 50
    guidance_scale: float = 0.0
    seed: int | None = None

    # 潜变量
    latents: torch.Tensor | None = None
    height: int | None = None
    width: int | None = None

    # KV 缓存传输（Bagel 模型）
    past_key_values: Any | None = None
    kv_metadata: dict[str, Any] | None = None

    # CFG 多 KV
    cfg_text_past_key_values: Any | None = None
    cfg_img_past_key_values: Any | None = None

    # LoRA
    lora_request: LoRARequest | None = None
    lora_scale: float = 1.0

    # 轨迹记录
    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False

    # ... 更多参数
```

这是一个大型数据类，包含扩散推理所需的所有参数，包括：
- 推理步数、引导尺度、种子
- 潜变量维度和张量
- KV 缓存传输参数
- LoRA 配置
- STA/VSA 优化参数
- 性能分析开关

### 类型别名

```python
OmniSingletonPrompt = str | list[int] | OmniTextPrompt | OmniTokensPrompt | OmniEmbedsPrompt
OmniPromptType = PromptType | OmniTextPrompt | OmniTokensPrompt | OmniEmbedsPrompt | OmniCustomPrompt
OmniSamplingParams = SamplingParams | OmniDiffusionSamplingParams
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniTextPrompt` | TypedDict | 扩展文本提示（含嵌入） |
| `OmniTokensPrompt` | TypedDict | 扩展 Token 提示（含嵌入） |
| `OmniTokenInputs` | TypedDict | 扩展 Token 输入（含嵌入） |
| `OmniEmbedsPrompt` | TypedDict | 扩展嵌入提示 |
| `OmniCustomPrompt` | TypedDict | 自定义扩散提示 |
| `token_inputs_omni` | 函数 | 构造 OmniTokenInputs |
| `OmniDiffusionSamplingParams` | 数据类 | 扩散模型采样参数 |
| `OmniPromptType` | 类型别名 | 统一的提示类型 |
| `OmniSamplingParams` | 类型别名 | 统一的采样参数类型 |

## 与其他模块的关系

- 扩展 `vllm.inputs.data` 中的基础类型
- 被 `preprocess.py` 中的 `OmniInputPreprocessor` 处理
- `OmniTokensPrompt` 被 `patch.py` 替换到 vLLM 中
- `OmniPromptType` 被 `outputs.py` 引用
- `OmniDiffusionSamplingParams` 被扩散模型执行器使用
- 引用 `lora.request.LoRARequest` 用于 LoRA 支持

## 总结

`data.py` 是 vllm-omni 输入系统的类型基础，通过 TypedDict 继承保持了与 vLLM 原始类型的兼容性（因为 TypedDict 本质是 dict），同时添加了多模态推理所需的扩展字段。`OmniDiffusionSamplingParams` 集中管理了扩散推理的所有参数，简化了参数传递。
