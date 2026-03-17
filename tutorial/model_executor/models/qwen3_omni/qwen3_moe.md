# `qwen3_moe.py` — MoE 稀疏专家模型

## 文件概述

本文件为 Qwen3 Omni Talker 提供 MoE（混合专家）层实现。包含两种 MoE 实现：`Qwen3OmniMoeSparseMoeBlock`（使用独立专家 MLP）和对 vLLM 上游 `Qwen3MoeForCausalLM` 的薄包装。

## 关键代码解析

### 1. 独立专家 MoE 块

```python
class Qwen3OmniMoeSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        self.experts = nn.ModuleList([
            Qwen3MoeMLP(...) for i in range(self.num_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, ...)
```

不使用 `FusedMoE` 算子，而是创建独立的 MLP 专家实例，适用于不支持融合 MoE 的平台。

### 2. 路由与专家选择

```python
def _route_tokens(self, router_logits):
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
```

使用 softmax + top-k 选择路由，可选归一化路由权重。

### 3. 专家前向传播

```python
def _forward_experts(self, hidden_states, selected_experts, routing_weights):
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2,1,0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1,-2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
        current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states)
```

按需激活专家，仅处理被路由到的 token。

### 4. Qwen3MoeForCausalLM 包装

```python
class Qwen3MoeForCausalLM(_BaseQwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)  # 不调用 super().__init__() 避免重复注册
        self.model = Qwen3MoeModel(vllm_config=vllm_config, ...)
```

薄包装层，提取 MoE 层信息（`num_moe_layers`, `num_logical_experts` 等）。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `Qwen3OmniMoeSparseMoeBlock` | 类 | 独立专家 MoE 块 |
| `Qwen3MoeForCausalLM` | 类 | MoE 因果语言模型包装 |
| `_route_tokens()` | 方法 | Top-k 专家路由 |
| `_forward_experts()` | 方法 | 按需专家前向传播 |

## 与其他模块的关系

- **被引用**: `qwen3_omni_moe_talker.py` 中的 `Qwen3OmniMoeModel` 继承此处的 `Qwen3MoeForCausalLM`
- **依赖**: vLLM 上游 `Qwen3MoeDecoderLayer`、`Qwen3MoeMLP`、`FusedMoE`

## 总结

`qwen3_moe.py` 提供了两种 MoE 实现路径：融合 MoE（高性能）和独立专家（兼容性好），并通过薄包装层暴露 MoE 超参数供上层调度使用。
