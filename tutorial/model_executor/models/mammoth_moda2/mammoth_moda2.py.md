# `mammoth_moda2.py` — MammothModa2 核心模型实现

## 文件概述

MammothModa2 的核心实现文件，约 827 行。包含 MoE 路由函数、MoE 解码层、双词表 LLM、AR 模型（继承 Qwen2.5-VL）、T2I token 约束逻辑、以及顶层多阶段路由。

## 关键代码解析

### 1. MoE 路由函数

```python
def moe_forward(hidden_states, und_expert, gen_expert, gen_token_mask=None):
    """按 mask 将 token 路由到不同专家"""
    if gen_expert is None or not gen_token_mask.any():
        return und_expert(hidden_states)
    if gen_token_mask.all():
        return gen_expert(hidden_states)
    # 混合批次：分离→分别计算→重排合并
    gen_pos = torch.where(flat_mask)[0]
    und_pos = torch.where(~flat_mask)[0]
    gen_out = gen_expert(gen_hid)
    und_out = und_expert(und_hid)
    merged = torch.cat([gen_out, und_out])[inverse_order]
```

### 2. MoE 解码层

```python
class Mammoth2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx, ...):
        if moe_enable(config.moe_type, "ffn", layer_idx):
            self.gen_mlp = Qwen2MLP(...)  # 生成专家
    def forward(self, positions, hidden_states, residual, gen_token_mask=None):
        # 注意力：共享（不分路由）
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        # FFN：MoE 路由
        hidden_states = moe_forward(hidden_states, self.mlp, self.gen_mlp, gen_token_mask)
```

### 3. 双词表 LLM

```python
class MammothModa2Qwen2ForCausalLM(nn.Module):
    def __init__(self, ...):
        self.embed_tokens = VocabParallelEmbedding(base_vocab_size, ...)
        self.gen_embed_tokens = VocabParallelEmbedding(gen_vocab_size, ...)
        self.lm_head = ParallelLMHead(base_vocab_size, ...)
        self.gen_head = ParallelLMHead(gen_vocab_size, ...)

    def compute_logits(self, hidden_states):
        base_logits = self.logits_processor(self.lm_head, hidden_states)
        gen_logits = self.gen_logits_processor(self.gen_head, hidden_states)
        return torch.cat([base_logits, gen_logits], dim=-1)
```

### 4. T2I token 约束

```python
def _apply_t2i_token_constraints(self, logits):
    column_id = generated_len % (ar_width + 1)
    if column_id == ar_width:
        # 行末：只允许 EOL token
        row.fill_(neg_inf)
        row[eol_token_id] = eol_logit
    else:
        # 行内：只允许视觉 token
        row[:visual_start] = neg_inf
        row[visual_end + 1:] = neg_inf
```

### 5. 顶层多阶段路由

```python
class MammothModa2ForConditionalGeneration(nn.Module):
    def __init__(self, ...):
        if self.model_stage == "ar":
            self.ar = init_vllm_registered_model(architectures=["MammothModa2ARForConditionalGeneration"])
        elif self.model_stage == "dit":
            self.dit = init_vllm_registered_model(architectures=["MammothModa2DiTPipeline"])
```

## 核心类/函数

| 名称 | 类型 | 描述 |
|------|------|------|
| `moe_enable()` | 函数 | 判断指定层是否启用 MoE |
| `moe_forward()` | 函数 | MoE 路由 + 前向计算 |
| `Mammothmoda2Processor` | 类 | MammothU tokenizer 的处理器 |
| `MammothModa2ARProcessingInfo` | 类 | 处理信息（返回 VL 子配置） |
| `Mammoth2DecoderLayer` | 类 | MoE Qwen2 解码层 |
| `MammothModa2Qwen2ForCausalLM` | 类 | 双词表 MoE LLM |
| `MammothModa2ARForConditionalGeneration` | 类 | AR 阶段（继承 Qwen2.5-VL） |
| `MammothModa2ForConditionalGeneration` | 类 | 顶层多阶段路由 |

## 与其他模块的关系

- 继承 `Qwen2_5_VLForConditionalGeneration`（视觉编码、多模态处理）
- 使用 `Qwen2DecoderLayer`、`Qwen2MLP` 作为基础组件
- DiT 阶段委托给 `pipeline_mammothmoda2_dit.py`
- 权重映射使用 `WeightsMapper` 处理 `llm_model.*` 前缀

## 总结

MammothModa2 的核心创新在于：(1) 在 Qwen2 解码层中引入 MoE FFN 路由，分离理解和生成能力；(2) 双词表设计使文本和图像 token 各自拥有独立的嵌入和输出空间；(3) 行级 T2I 约束确保图像 token 生成的结构正确性。
