# `hsdp.py` -- HSDP 混合分片数据并行

## 文件概述

`hsdp.py` 实现了 HSDP（Hybrid Sharded Data Parallelism）推理支持，将 PyTorch 的 `fully_shard` (FSDP2) API 集成到 vllm-omni 中。HSDP 在 2D 网格上组合了参数分片和副本复制，适用于大模型的多 GPU 推理。

## 关键代码解析

### HSDPInferenceConfig 配置

```python
@dataclass
class HSDPInferenceConfig:
    enabled: bool = False
    hsdp_replicate_size: int = 1        # 副本组大小
    hsdp_shard_size: int = -1           # 分片组大小（-1 = 自动）
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype | None = None
    reshard_after_forward: bool = True  # forward 后是否重新分片
```

HSDP 2D 网格的两个维度：
- **replicate 维度**: 持有相同参数分片的 rank 组（训练时用于梯度 all-reduce）
- **shard 维度**: 每个 rank 持有不同参数分片的组（推理时用于参数 all-gather）

### _create_hsdp_mesh -- 创建 2D 设备网格

```python
def _create_hsdp_mesh(device_type, replicate_size, shard_pg):
    shard_size = torch.distributed.get_world_size(shard_pg)
    world_size = replicate_size * shard_size
    mesh_tensor = torch.arange(world_size).reshape(replicate_size, shard_size)
    device_mesh = init_device_mesh(
        device_type,
        mesh_shape=(replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    return device_mesh
```

### apply_hsdp_to_model -- 应用 HSDP 分片

```python
def apply_hsdp_to_model(model, hsdp_config):
    world_group = get_world_group()
    fs_group = get_fs_group()

    # 验证 FS 组与 HSDP 分片大小匹配
    if fs_world_size != hsdp_shard_size:
        raise ValueError(...)

    # 创建混合精度策略
    mp_policy = MixedPrecisionPolicy(
        param_dtype=hsdp_config.param_dtype,
        reduce_dtype=hsdp_config.reduce_dtype,
    )

    # 创建 2D 设备网格
    device_mesh = _create_hsdp_mesh(device_type, replicate_size, fs_group.device_group)

    # 应用分片
    shard_model(model, mesh=device_mesh, mp_policy=mp_policy,
                hsdp_shard_conditions=model._hsdp_shard_conditions)

    # 推理模式：禁用梯度
    for param in model.parameters():
        param.requires_grad = False
    return model
```

### shard_model -- 模型分片

```python
def shard_model(model, *, reshard_after_forward, mp_policy, mesh, hsdp_shard_conditions):
    num_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if any(cond(name, module) for cond in hsdp_shard_conditions):
            fully_shard(module, **hsdp_kwargs)
            num_sharded += 1
    fully_shard(model, **hsdp_kwargs)  # 分片根模块
```

分片策略由模型定义的 `_hsdp_shard_conditions` 决定，每个条件是一个 `(name, module) -> bool` 的可调用对象。逆序遍历确保子模块先于父模块被分片。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `HSDPInferenceConfig` | dataclass | HSDP 推理配置（分片/副本大小、精度策略） |
| `_create_hsdp_mesh()` | 函数 | 创建 2D DeviceMesh |
| `apply_hsdp_to_model()` | 函数 | 将 HSDP 分片应用到模型 |
| `shard_model()` | 函数 | 基于条件列表对模型子模块执行 fully_shard |

## 与其他模块的关系

- **parallel_state.py**: 使用 `get_world_group()`, `get_fs_group()` 获取分布式组信息
- **模型定义**: 模型需要定义 `_hsdp_shard_conditions` 属性
- **PyTorch FSDP2**: 底层使用 `torch.distributed.fsdp.fully_shard`

## 总结

`hsdp.py` 将 PyTorch 的 FSDP2 API 适配到 vllm-omni 的分布式框架中。通过 `_hsdp_shard_conditions` 机制，模型可以声明哪些子模块应该被分片，框架自动处理设备网格创建和混合精度配置。推理模式下禁用梯度以节省内存。该模块使得大型扩散模型可以在参数超过单 GPU 显存的情况下进行推理。
