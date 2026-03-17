# `mammoth_moda2.py` — MammothModa2 配置类定义与注册

## 文件概述

该文件定义了 MammothModa2 多模态模型的完整配置类层次，并将其注册到 HuggingFace `AutoConfig`。MammothModa2 基于 Qwen2.5-VL 架构，扩展了生成词汇表（gen vocab）和 MoE（Mixture of Experts）等特性。

## 关键代码解析

### 配置层次结构

```
Mammothmoda2Config (顶层组合配置)
└── llm_config: Mammothmoda2Qwen2_5_VLConfig (VL 配置)
    ├── text_config: Mammothmoda2Qwen2_5_VLTextConfig (文本配置)
    └── vision_config: Mammothmoda2Qwen2_5_VLVisionConfig (视觉配置)
```

### 视觉配置

```python
class Mammothmoda2Qwen2_5_VLVisionConfig(Qwen2_5_VLVisionConfig):
    model_type = "mammothmoda2_qwen2_5_vl_vision"
    # 默认值与 Qwen2.5-VL 一致，但 fullatt_block_indexes 默认为 [7, 15, 23, 31]
```

继承 Qwen2.5-VL 的视觉配置，仅修改 `model_type`。

### 文本配置（核心扩展）

```python
class Mammothmoda2Qwen2_5_VLTextConfig(Qwen2_5_VLTextConfig):
    model_type = "mammothmoda2_qwen2_5_vl_text"

    def __init__(self, ...,
                 extra_gen_vocab: bool = True,
                 gen_vocab_size: int = 32800,
                 gen_vocab_start_index: int | None = None,
                 moe_type: str = "ffn", ...):
        # ...
        if gen_vocab_start_index is None:
            self.gen_vocab_start_index = (
                self.vocab_size if self.extra_gen_vocab
                else self.vocab_size - self.gen_vocab_size
            )

        # 扩展 vocab_size 以覆盖 gen vocab 范围
        if self.extra_gen_vocab:
            self.vocab_size = int(self.gen_vocab_start_index) + int(self.gen_vocab_size)
```

关键扩展：
- **gen_vocab（生成词汇表）**: 额外的 32800 个 token 用于图像/音频生成，独立的 embedding 和 head
- **moe_type**: MoE 类型配置
- **vocab_size 调整**: 当 `extra_gen_vocab=True` 时，扩展总词汇表大小以覆盖生成 token ID 范围

### VL 组合配置

```python
class Mammothmoda2Qwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "mammothmoda2_qwen2_5_vl"
    sub_configs = {
        "vision_config": Mammothmoda2Qwen2_5_VLVisionConfig,
        "text_config": Mammothmoda2Qwen2_5_VLTextConfig,
    }

    def __init__(self, text_config=None, vision_config=None, ...):
        # 将 gen vocab 参数透传到 text_config
        text_extra_kwargs = {
            "extra_gen_vocab": extra_gen_vocab,
            "gen_vocab_size": gen_vocab_size,
            "moe_type": moe_type,
        }
        # ...
        self.tokenizer_class = "MammothUTokenizer"
```

### 顶层组合配置

```python
class Mammothmoda2Config(PretrainedConfig):
    model_type = "mammothmoda2"
    is_composition = True
    sub_configs: ClassVar = {"llm_config": AutoConfig}

    def __init__(self, *, llm_config=None, gen_vae_config=None,
                 gen_dit_config=None, gen_condition_mode="image", ...):
        self.llm_config = AutoConfig.for_model(**llm_config) if llm_config else None
        self.gen_vae_config = gen_vae_config
        self.gen_dit_config = gen_dit_config
        self.gen_condition_mode = gen_condition_mode  # "text" | "image" | "text_image"
        self.gen_axes_dim_rope = gen_axes_dim_rope or [40, 40, 40]
```

顶层配置包含：
- **llm_config**: 语言模型配置（通过 AutoConfig 递归解析）
- **gen_vae_config**: 生成用 VAE 配置
- **gen_dit_config**: 生成用 DiT（Diffusion Transformer）配置
- **gen_condition_mode**: 生成条件模式
- **gen_transport_config**: 传输配置

### 代理属性

```python
class Mammothmoda2Config:
    @property
    def vision_config(self):
        return self._require_llm_config().vision_config

    @property
    def image_token_id(self) -> int:
        return int(self._require_llm_config().image_token_id)
    # ...
```

顶层配置通过属性代理将 vLLM 需要的视觉相关字段透传到嵌套的 `llm_config`，使多模态处理代码能直接在顶层配置上访问这些字段。

### AutoConfig 注册

```python
AutoConfig.register(Mammothmoda2Config.model_type, Mammothmoda2Config)
AutoConfig.register(Mammothmoda2Qwen2_5_VLConfig.model_type, Mammothmoda2Qwen2_5_VLConfig)
AutoConfig.register(Mammothmoda2Qwen2_5_VLTextConfig.model_type, Mammothmoda2Qwen2_5_VLTextConfig)
AutoConfig.register(Mammothmoda2Qwen2_5_VLVisionConfig.model_type, Mammothmoda2Qwen2_5_VLVisionConfig)
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Mammothmoda2Qwen2_5_VLVisionConfig` | 配置类 | 视觉编码器配置 |
| `Mammothmoda2Qwen2_5_VLTextConfig` | 配置类 | 文本模型配置（含 gen vocab 扩展） |
| `Mammothmoda2Qwen2_5_VLConfig` | 配置类 | VL 组合配置（text + vision） |
| `Mammothmoda2Config` | 配置类 | 顶层组合配置（LLM + VAE + DiT） |

## 与其他模块的关系

- **与分词器配合**: 指定 `tokenizer_class = "MammothUTokenizer"`，对应 `tokenizers/mammoth_moda2_tokenizer.py`。
- **继承 Qwen2.5-VL**: 视觉和文本配置继承自 HuggingFace 的 Qwen2.5-VL 配置类。
- **被 configs/__init__.py 管理**: 通过延迟加载和即时导入两种方式暴露。
- **服务模型加载**: `AutoConfig.from_pretrained()` 可通过 `model_type` 自动找到正确的配置类。

## 总结

该文件定义了 MammothModa2 的四层配置类层次，在 Qwen2.5-VL 基础上扩展了生成词汇表（用于图像/音频生成的额外 token 空间）、MoE 支持、DiT/VAE 生成模块配置等多模态生成能力。通过 AutoConfig 注册和属性代理，确保与 HuggingFace 和 vLLM 生态的无缝兼容。
