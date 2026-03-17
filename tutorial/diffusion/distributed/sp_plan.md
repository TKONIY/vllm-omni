# `sp_plan.py` -- 序列并行配置与计划类型定义

## 文件概述

`sp_plan.py` 定义了序列并行（Sequence Parallelism, SP）的配置类和计划类型系统。通过声明式的 `_sp_plan` 字典，模型可以指定如何在 forward 过程中自动分片和聚合张量，无需修改 forward 方法。该模块对应 diffusers 库中的 Context Parallelism 概念。

## 关键代码解析

### SequenceParallelConfig -- SP 配置

```python
@dataclass
class SequenceParallelConfig:
    ulysses_degree: int = 1    # Ulysses (All-to-All) 并行度
    ring_degree: int = 1       # Ring 注意力并行度
    convert_to_fp32: bool = True  # Ring 注意力输出是否转 FP32

    @property
    def sequence_parallel_size(self) -> int:
        return self.ulysses_degree * self.ring_degree

    def get_world_size(self) -> int:
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size()

    def setup(self, rank, world_size, device):
        """运行时初始化，验证配置与实际并行组匹配。"""
        self._rank = rank
        self._world_size = world_size
        expected_sp_size = self.ulysses_degree * self.ring_degree
        if expected_sp_size != self.get_world_size():
            raise ValueError(...)
```

### SequenceParallelInput -- 输入分片配置

```python
@dataclass(frozen=True)
class SequenceParallelInput:
    split_dim: int                    # 分片维度
    expected_dims: int | None = None  # 期望的张量维度数（用于验证）
    split_output: bool = False        # True 时分片输出而非输入
    auto_pad: bool = False            # 自动填充使长度可被 world_size 整除
```

使用示例：
```python
# 在序列维度（dim=1）上分片 hidden_states
SequenceParallelInput(split_dim=1, expected_dims=3)

# 分片 RoPE 输出
SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True)

# 可变长度序列的自动填充分片
SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True)
```

### SequenceParallelOutput -- 输出聚合配置

```python
@dataclass(frozen=True)
class SequenceParallelOutput:
    gather_dim: int                   # 聚合维度
    expected_dims: int | None = None  # 期望的张量维度数
```

### SequenceParallelPartialInput -- 部分分片配置

```python
@dataclass(frozen=True)
class SequenceParallelPartialInput:
    split_dim: int
    text_len_source: str | int    # 文本长度来源（参数名或固定值）
    expected_dims: int | None = None
    split_output: bool = False
```

专为双流模型设计（如 Qwen），其中文本部分需要保持完整（用于联合注意力），只分片图像部分。

### _sp_plan 声明式计划

```python
# 完整的 _sp_plan 示例
class MyTransformer(nn.Module):
    _sp_plan = {
        # 根层级：分片模型输入
        "": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
            "encoder_hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # 子模块：分片 RoPE 输出
        "pos_embed": {
            0: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        # 子模块：聚合 proj_out 输出
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

计划中的键（Key）含义：
- `""`: 根模块（模型本身）
- `"module_name"`: 指定子模块
- `"module_name.*"`: ModuleList 的所有子模块

### validate_sp_plan -- 计划验证

```python
def validate_sp_plan(plan):
    """验证 _sp_plan 字典的正确性。"""
    for module_id, module_plan in plan.items():
        # 检查键类型
        # 检查值类型（输入/输出配置）
        # 检查整数键必须有 split_output=True
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `SequenceParallelConfig` | dataclass | SP 配置（Ulysses + Ring 参数） |
| `SequenceParallelInput` | dataclass | 输入张量分片配置 |
| `SequenceParallelOutput` | dataclass | 输出张量聚合配置 |
| `SequenceParallelPartialInput` | dataclass | 部分分片配置（双流模型用） |
| `SequenceParallelModelPlan` | 类型别名 | `_sp_plan` 字典类型 |
| `validate_sp_plan()` | 函数 | 验证 `_sp_plan` 正确性 |
| `get_sp_plan_from_model()` | 函数 | 从模型获取并验证 `_sp_plan` |

## 与其他模块的关系

- **parallel_state.py**: `SequenceParallelConfig` 通过该模块获取运行时并行状态
- **sp_sharding.py**: `_sp_plan` 中声明的操作最终由分片工具函数执行
- **模型定义**: 模型通过定义 `_sp_plan` 类属性来声明 SP 行为

## 总结

`sp_plan.py` 实现了一套声明式的序列并行配置系统。模型开发者只需通过 `_sp_plan` 字典声明哪些张量需要分片/聚合，框架自动通过 Hook 处理通信。`SequenceParallelPartialInput` 是 vllm-omni 针对双流注意力模型的特有扩展，解决了"文本部分保持完整、图像部分分片"的需求。整套类型系统与 diffusers 的 Context Parallelism 兼容但更灵活。
