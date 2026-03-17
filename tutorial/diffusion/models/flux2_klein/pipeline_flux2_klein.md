# `pipeline_flux2_klein.py` -- Flux 2 Klein 推理管线

## 文件概述

实现 `Flux2KleinPipeline`，Flux 2 Klein 的完整推理管线。使用 Qwen3 作为文本编码器，支持 CFG 并行和序列并行。与 Flux2Pipeline 使用 Mistral3 不同，Klein 版本使用 Qwen3ForCausalLM + Qwen2TokenizerFast。

**文件路径**: `vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py`

## 关键代码解析

### 管线初始化

```python
class Flux2KleinPipeline(nn.Module, CFGParallelMixin):
    def __init__(self, *, od_config):
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(...)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(...)
        self.vae = AutoencoderKLFlux2.from_pretrained(...)
        self.transformer = Flux2Transformer2DModel(od_config=od_config, ...)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(...)
```

### 文本编码

Klein 使用 Qwen3 的隐藏状态作为文本条件嵌入，而非独立的 CLIP+T5 编码器。

### 去噪循环

标准的 Flow Match Euler 去噪，支持 CFG 并行和可选的图像编辑（i2i）模式。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Flux2KleinPipeline` | nn.Module | Klein 完整推理管线 |
| `get_flux2_klein_post_process_func` | 工厂函数 | 图像后处理函数 |

## 与其他模块的关系

| 关系 | 模块 | 说明 |
|------|------|------|
| 依赖 | `flux2_klein_transformer.py` | SP 版 Transformer |
| 混入 | `CFGParallelMixin` | CFG 并行 |
| 对比 | `flux2/pipeline_flux2.py` | Mistral3 编码器版 |

## 总结

`Flux2KleinPipeline` 使用 Qwen3 作为文本编码器，配合支持序列并行的 Flux 2 Transformer，适用于大规模高分辨率图像生成。CFG 并行和序列并行可同时使用，实现最大化的 GPU 利用效率。
