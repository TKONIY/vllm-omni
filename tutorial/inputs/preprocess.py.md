# `inputs/preprocess.py` — 输入预处理器

## 文件概述

`preprocess.py` 定义了 `OmniInputPreprocessor`，扩展 vLLM 的 `InputPreprocessor` 以处理 omni 特有的输入类型。核心任务是将各种格式的用户输入（文本、token、嵌入）转换为模型可消费的标准化输入。

## 关键代码解析

### _process_text — 文本输入处理

```python
def _process_text(self, parsed_content, tokenization_kwargs=None):
    prompt_text = parsed_content["prompt"]
    mm_processor_kwargs = parsed_content.get("mm_processor_kwargs") or {}

    if multi_modal_data := parsed_content.get("multi_modal_data"):
        # 多模态文本输入：调用基类多模态处理
        inputs = self._process_multimodal(prompt_text, multi_modal_data, mm_processor_kwargs, ...)
    elif mm_processor_kwargs:
        # 有处理器参数但无多模态数据（如 GLM-Image 的文生图）
        inputs = self._process_multimodal(prompt_text, {}, mm_processor_kwargs, ...)
    else:
        # 纯文本输入
        prompt_token_ids = self._tokenize_prompt(prompt_text, ...)
        inputs = token_inputs_omni(prompt_token_ids, ...)

    # 注入 omni 扩展字段
    if prompt_embeds := parsed_content.get("prompt_embeds"):
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information := parsed_content.get("additional_information"):
        inputs["additional_information"] = additional_information
```

关键改进：支持 `mm_processor_kwargs` 独立使用（无需 `multi_modal_data`），这是 GLM-Image 等文生图模型所需的。

### _process_tokens — Token 输入处理

```python
def _process_tokens(self, parsed_content, tokenization_kwargs=None):
    prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], ...)
    prompt_embeds = parsed_content.get("prompt_embeds")
    additional_information = parsed_content.get("additional_information")

    if multi_modal_data := parsed_content.get("multi_modal_data"):
        inputs = self._process_multimodal(prompt_token_ids, multi_modal_data, ...)
    else:
        inputs = token_inputs_omni(
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            additional_information=additional_information,
        )

    # 注入扩展字段
    if prompt_embeds is not None:
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information is not None:
        inputs["additional_information"] = additional_information
```

### _process_embeds — 嵌入输入处理

```python
def _process_embeds(self, parsed_content):
    inputs = super()._process_embeds(parsed_content)
    # 注入 omni 特有的附加信息
    if additional_information := parsed_content.get("additional_information"):
        inputs["additional_information"] = additional_information
    return inputs
```

### _prompt_to_llm_inputs — 统一入口

```python
def _prompt_to_llm_inputs(self, prompt, tokenization_kwargs=None):
    if "prompt_embeds" in prompt:
        return self._process_embeds(prompt)
    if "prompt_token_ids" in prompt:
        return self._process_tokens(prompt)
    if "prompt" in prompt:
        return self._process_text(prompt, tokenization_kwargs=tokenization_kwargs)
    assert_never(prompt)
```

根据输入字典中的键分发到对应的处理方法。优先级：嵌入 > Token > 文本。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniInputPreprocessor` | 类 | Omni 输入预处理器 |
| `_process_text` | 方法 | 处理文本输入（支持无数据的 mm_processor_kwargs） |
| `_process_tokens` | 方法 | 处理 Token 输入（注入嵌入和附加信息） |
| `_process_embeds` | 方法 | 处理嵌入输入（注入附加信息） |
| `_prompt_to_llm_inputs` | 方法 | 统一的输入分发入口 |

## 与其他模块的关系

- 继承 `vllm.inputs.preprocess.InputPreprocessor`
- 使用 `data.py` 中定义的 `OmniTextPrompt`、`OmniTokensPrompt`、`OmniEmbedsPrompt`
- 调用 `token_inputs_omni()` 构造标准化输出
- 被引擎层在请求预处理阶段调用

## 总结

`OmniInputPreprocessor` 是用户输入与模型执行之间的转换层，通过扩展 vLLM 的预处理逻辑，确保 prompt 嵌入和附加信息能正确传递到后续的调度和推理阶段。其设计保持了与 vLLM 输入系统的完全兼容性。
