# MammothModa2 模型模块架构概览

## 模块简介

MammothModa2 是一个多模态理解与图像生成模型，基于 Qwen2.5-VL 架构扩展了 MoE（Mixture-of-Experts）路由机制和图像生成（t2i）token 约束。支持 AR（自回归）和 DiT（扩散 Transformer）两个推理阶段。

## 架构图

```
输入（文本 + 图像）
       │
       ▼
┌──────────────────────────────────┐
│  MammothModa2ForConditional      │  ← 顶层路由
│  Generation                      │
│  ├── AR 阶段                     │
│  │   └── MammothModa2AR          │
│  │       ├── Qwen2.5-VL 视觉    │  ← 图像编码
│  │       └── MammothModa2Qwen2   │  ← MoE LLM
│  │           ├── embed_tokens     │  ← 基础词嵌入
│  │           ├── gen_embed_tokens │  ← 生成词嵌入
│  │           ├── Mammoth2Decoder  │  ← MoE 解码层
│  │           │   ├── self_attn    │  ← 共享注意力
│  │           │   ├── mlp          │  ← 理解专家
│  │           │   └── gen_mlp      │  ← 生成专家
│  │           ├── lm_head          │  ← 基础 logits
│  │           └── gen_head         │  ← 生成 logits
│  └── DiT 阶段                    │
│      └── MammothModa2DiTPipeline │  ← 扩散图像生成
└──────────────────────────────────┘
```

## 文件列表

| 文件 | 描述 |
|------|------|
| `__init__.py` | 模块入口，注册 tokenizer |
| `mammoth_moda2.py` | 核心实现：MoE 路由、AR 模型、token 约束 |
| `pipeline_mammothmoda2_dit.py` | DiT 管线兼容性 shim |

## 核心设计思想

1. **MoE FFN 路由**：在指定层范围内，FFN 分为理解专家（mlp）和生成专家（gen_mlp），通过 `gen_token_mask` 路由。

2. **双词表架构**：基础词汇（文本）和生成词汇（图像 token）使用独立的嵌入表和输出头，通过 `gen_vocab_start_index` 区分。

3. **T2I token 约束**：在 `compute_logits` 后应用行级约束——行内只允许视觉 token，行末强制 EOL token。

4. **多阶段部署**：AR 和 DiT 阶段通过 `model_stage` 参数在同一 `MammothModa2ForConditionalGeneration` 中路由。
