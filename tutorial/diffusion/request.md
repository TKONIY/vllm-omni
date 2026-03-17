# `request.py` — 扩散请求数据结构

## 文件概述

`request.py` 定义了扩散推理请求的数据结构 `OmniDiffusionRequest`，封装了提示词列表、采样参数和请求标识。它是 DiffusionEngine 与执行器之间的标准通信格式。

## 关键代码解析

### OmniDiffusionRequest

```python
@dataclass
class OmniDiffusionRequest:
    prompts: list[OmniPromptType]
    sampling_params: OmniDiffusionSamplingParams
    request_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        # 根据 guidance_scale 和 negative_prompt 自动设置 do_classifier_free_guidance
        if self.sampling_params.guidance_scale > 1.0 and any(
            (not isinstance(p, str) and p.get("negative_prompt")) for p in self.prompts
        ):
            self.sampling_params.do_classifier_free_guidance = True

        # 设置 guidance_scale_2 默认值
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

        # 处理 guidance_scale 的零值情况
        if self.sampling_params.guidance_scale:
            self.sampling_params.guidance_scale_provided = True
        else:
            self.sampling_params.guidance_scale = 1.0
```

`__post_init__` 中的关键逻辑：
1. **CFG 自动检测**：当 `guidance_scale > 1.0` 且存在 negative prompt 时，自动开启 Classifier-Free Guidance
2. **双引导尺度**：`guidance_scale_2` 默认与 `guidance_scale` 相同
3. **零值处理**：guidance_scale 为 0 时重置为 1.0 并标记为未提供

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `OmniDiffusionRequest` | dataclass | 扩散推理请求，包含提示词、采样参数和请求 ID |

## 与其他模块的关系

- 被 `diffusion_engine.py` 的 `step()` 方法创建和传递
- 通过 `scheduler.py` 发送到 worker 进程
- 被 `worker/diffusion_model_runner.py` 的 `execute_model` 方法消费
- 依赖 `vllm_omni.inputs.data` 中的 `OmniDiffusionSamplingParams` 和 `OmniPromptType`

## 总结

`OmniDiffusionRequest` 是一个简洁的请求封装类，将提示词和采样参数打包为统一格式。它在 `__post_init__` 中自动处理 CFG 配置等常见参数推导，减少了上层代码的配置负担。
