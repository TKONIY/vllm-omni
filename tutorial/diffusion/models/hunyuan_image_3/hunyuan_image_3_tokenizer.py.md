# `hunyuan_image_3_tokenizer.py` — HunyuanImage3 多模态 Tokenizer

## 文件概述

本文件实现了 HunyuanImage3 的多模态 Tokenizer 封装 `TokenizerWrapper`，负责将文本、图像和控制信号统一编码为模型可消费的 token 序列。它处理了 HunyuanImage3 独特的序列格式：将生成图像 token、条件图像 token、时间步 token、引导 token 和文本 token 混合在同一序列中。

## 关键代码解析

### 1. TokenizerEncodeOutput

```python
class TokenizerEncodeOutput(BaseOutput):
    tokens: torch.Tensor | None = None
    timestep_scatter_index: torch.Tensor | None = None
    guidance_scatter_index: torch.Tensor | None = None
    text_slices: list[slice] | None = None
    gen_image_slices: list[slice] | None = None
    joint_image_slices: list[slice] | None = None
    cond_vae_image_slices: list[slice] | None = None
    cond_vit_image_slices: list[slice] | None = None
    text_mask: torch.Tensor | None = None
    gen_image_mask: torch.Tensor | None = None
    ...
```

输出数据类包含完整的 token 序列及各种位置索引和掩码，使管线能精确定位每种模态在序列中的位置。

### 2. TokenizerWrapper

```python
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<boi>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<eoi>")
        self.img_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.cfg_token_id = self.tokenizer.convert_tokens_to_ids("<cfg>")
        self.ratio_token_offset = self.tokenizer.convert_tokens_to_ids("<img_ratio_0>")
```

初始化时解析所有特殊 token ID，包括 `<boi>`（图像开始）、`<eoi>`（图像结束）、`<img>`（图像占位符）、`<cfg>`（CFG 引导标记）和 `<img_ratio_N>`（图像宽高比标记）。

### 3. 文本编码

```python
def encode_text(self, *texts, uncond_enabled=None, uncond_p=None, max_length=None, ...):
    # 支持批量文本编码
    # uncond_enabled 控制是否进行无条件训练（置零文本 token）
    # 始终在文本前添加 <bos> token
```

### 4. 聊天模板应用

```python
def apply_chat_template(self, batch_prompt=None, batch_message_list=None, mode="gen_image", ...):
    # 将提示/消息列表转换为完整的 token 序列
    # 处理文本区段、生成图像区段、条件图像区段的排列
    # 构建 attention mask、position embeddings 等
```

聊天模板是 HunyuanImage3 的核心功能之一，它将多轮对话、图像生成指令和条件图像统一编排为自回归序列格式。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `TokenizerEncodeOutput` | 数据类 | Tokenizer 输出：token 序列 + 各种索引/掩码 |
| `Conversation` | 类 | 对话格式定义 |
| `TokenizerWrapper` | 类 | 多模态 Tokenizer 封装 |
| `TokenizerWrapper.encode_text` | 方法 | 文本编码 |
| `TokenizerWrapper.pad` | 方法 | 序列填充 |
| `TokenizerWrapper.apply_chat_template` | 方法 | 聊天模板应用 |

## 与其他模块的关系

- **`pipeline_hunyuan_image_3.py`**：管线中通过 `TokenizerWrapper` 构建模型输入序列
- **`hunyuan_image_3_transformer.py`**：导入 `ImageInfo`、`JointImageInfo` 等数据类
- **transformers**：底层使用 `AutoTokenizer` 进行文本分词

## 总结

`hunyuan_image_3_tokenizer.py` 实现了 HunyuanImage3 独特的多模态序列编码逻辑，将文本、图像和控制信号统一编排为自回归格式的 token 序列。通过丰富的位置索引和掩码输出，使管线能精确控制每种模态在 Transformer 中的交互方式。
