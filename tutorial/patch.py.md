# `patch.py` — vLLM 猴子补丁系统

## 文件概述

`patch.py` 是 vllm-omni 与 vLLM 集成的关键文件，通过猴子补丁（monkey patching）机制，将 vLLM 中的核心类替换为 omni 扩展版本。该文件在 `__init__.py` 导入时自动执行。

## 关键代码解析

### GlmImageTextConfig 补丁

```python
try:
    from transformers.models.glm_image.configuration_glm_image import GlmImageTextConfig
    _original_glm_image_text_config_init = GlmImageTextConfig.__init__

    def _patched_glm_image_text_config_init(self, *args, **kwargs):
        _original_glm_image_text_config_init(self, *args, **kwargs)
        if self.rope_parameters is None:
            self.rope_parameters = {}
        if isinstance(self.rope_parameters, dict) and "mrope_section" not in self.rope_parameters:
            self.rope_parameters["mrope_section"] = [8, 12, 12]

    GlmImageTextConfig.__init__ = _patched_glm_image_text_config_init
except ImportError:
    pass
```

GLM-Image 使用 M-RoPE（多维旋转位置编码），但 transformers 库未在 `rope_parameters` 中暴露 `mrope_section`。此补丁确保 vLLM 能正确检测 M-RoPE。

### ModelConfig.is_mm_prefix_lm 补丁

```python
_orig_is_mm_prefix_lm = _ModelConfig.__dict__["is_mm_prefix_lm"].func

@_cached_property
def _patched_is_mm_prefix_lm(self) -> bool:
    return _orig_is_mm_prefix_lm(self) or getattr(self.hf_config, "model_type", None) == "bagel"

_ModelConfig.is_mm_prefix_lm = _patched_is_mm_prefix_lm
```

将 Bagel 模型加入多模态前缀 LM 的识别列表，使其获得与 Gemma3/Molmo2/PaliGemma 相同的双向注意力处理。

### RequestStatus 枚举扩展

```python
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)
```

添加 `WAITING_FOR_CHUNK` 状态，用于异步分块传输模式。值为 -1 以确保被视为"未完成"状态。

### 全局类替换

```python
for module_name, module in sys.modules.items():
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    # ... 更多替换
```

遍历所有已加载的 vLLM 模块，将以下核心类替换为 omni 扩展版本：

| 原始类 | 替换为 | 用途 |
|--------|--------|------|
| `EngineCoreOutput` | `OmniEngineCoreOutput` | 引擎核心输出 |
| `EngineCoreOutputs` | `OmniEngineCoreOutputs` | 引擎核心输出集合 |
| `TokensPrompt` | `OmniTokensPrompt` | Token 提示词 |
| `MRotaryEmbedding` | `OmniMRotaryEmbedding` | 旋转位置编码 |
| `Request` | `OmniRequest` | 请求对象 |
| `EngineCoreRequest` | `OmniEngineCoreRequest` | 引擎核心请求 |

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `_patched_glm_image_text_config_init` | 函数 | 修复 GLM-Image 的 M-RoPE 检测 |
| `_patched_is_mm_prefix_lm` | cached_property | 扩展多模态前缀 LM 识别 |
| 全局替换循环 | 代码块 | 将 vLLM 核心类替换为 omni 版本 |

## 与其他模块的关系

- 被 `__init__.py` 在包导入时触发
- 导入 `logger.py` 确保日志系统就绪
- 导入 `engine` 模块的 omni 版本类
- 导入 `inputs.data.OmniTokensPrompt` 替换 vLLM 的 `TokensPrompt`
- 导入 `request.OmniRequest` 替换 vLLM 的 `Request`

## 总结

`patch.py` 是 vllm-omni 的"手术刀"，通过精确的猴子补丁将 vLLM 的关键组件替换为支持多模态的版本。这种设计避免了分叉 vLLM 源码，使得 vllm-omni 可以作为插件层叠加在 vLLM 之上。
