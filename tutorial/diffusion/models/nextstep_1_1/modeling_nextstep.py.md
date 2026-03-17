# `modeling_nextstep.py` — NextStep-1.1 主模型

## 文件概述

该文件实现了 NextStep-1.1 的主模型 (`NextStepModel`)，基于 LLaMA 架构，集成了图像生成能力。模型使用 TP 感知的 LLaMA 解码器层，包含图像投影器和 Flow Matching Head，支持自回归图像 token 生成。

## 关键代码解析

### NextStepConfig

```python
class NextStepConfig(LlamaConfig):
    model_type = "nextstep"
    def __init__(self, ...,
        latent_size=32, latent_patch_size=2, latent_channels=16,
        boi=None, eoi=None, image_placeholder_id=None,
        fm_head_dim=1536, fm_head_layers=12,
    ):
```

扩展 LlamaConfig，添加 NextStep 特定的图像生成参数。

### 主模型结构

```python
class NextStepModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        # 图像投影器
        self.image_in_projector = nn.Linear(token_dim, config.hidden_size)
        self.image_out_projector = nn.Linear(config.hidden_size, config.hidden_size)
        # Flow Matching Head
        self.image_head = FlowMatchingHead(input_dim=token_dim, cond_dim=config.hidden_size, ...)
```

### Patchify / Unpatchify

```python
def patchify(self, img):
    # img: (bsz, C, H, W) -> tokens: (bsz, H*W/p^2, C*p^2)
    img = torch.einsum("nchpwq->nhwcpq", img.reshape(...))

def unpatchify(self, x, h=None, w=None):
    # tokens: (bsz, num_patches, C*p^2) -> img: (bsz, C, H, W)
    x = torch.einsum("nhwcpq->nchpwq", x.reshape(...))
```

将 latent 图像转换为 token 序列和反向操作。

### 输入嵌入准备

```python
def prepare_inputs_embeds(self, input_ids, latents=None):
    # 图像 token 使用 image_in_projector，文本 token 使用 embed_tokens
    im_indices = input_ids == self.config.image_placeholder_id
    inputs_embeds[im_indices] = image_embeds
    inputs_embeds[~im_indices] = token_embeds
```

### 权重加载（TP 感知）

```python
def load_weights(self, weights):
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `NextStepModel` | 类 | 基于 LLaMA 的主模型 |
| `NextStepConfig` | 类 | 模型配置，扩展 LlamaConfig |
| `get_2d_sincos_pos_embed` | 函数 | 2D 正弦余弦位置编码 |

## 与其他模块的关系

- 使用 `modeling_nextstep_llama.py` 中的 TP 感知 LLaMA 层
- 使用 `modeling_nextstep_heads.py` 中的 FlowMatchingHead
- 被 `pipeline_nextstep_1_1.py` 使用

## 总结

NextStepModel 将 LLaMA 架构改造为图像生成模型。文本 token 通过标准嵌入层处理，图像 token 通过 patchify + 投影器转换。每步自回归生成时，LLM 输出经过 image_out_projector 后送入 FlowMatchingHead 进行采样，生成下一个图像 token。
