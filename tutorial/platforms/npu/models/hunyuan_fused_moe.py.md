# `hunyuan_fused_moe.py` -- HunyuanFusedMoE Ascend NPU 实现

## 文件概述

该文件为 Hunyuan 图像生成模型的 FusedMoE（Mixture of Experts 融合算子）提供 Ascend NPU 专用实现。包含三部分：MC2 通信组初始化、MoE 通信方式选择、以及 `AscendHunyuanFusedMoE` 模型类。

## 关键代码解析

### 1. MC2 通信组初始化

```python
def _init_mc2_group_for_diffusion(
    world_size: int,
    data_parallel_size: int,
    tensor_parallel_size: int,
    backend: str,
    local_rank: int,
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size * tensor_parallel_size)
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks, local_rank, backend, group_name="mc2",
    )
```

MC2 是 Ascend NPU 上用于 MoE 专家间通信的通信组。该函数：
- 检查是否已初始化，避免重复创建
- 按 `data_parallel_size * tensor_parallel_size` 将所有 rank 分组
- 创建名为 `"mc2"` 的模型并行通信组

### 2. MoE 通信方式选择

```python
def _select_moe_comm_method(vllm_config: VllmConfig) -> MoECommType | None:
    soc_version = get_ascend_device_type()
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType.A5}:
        moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type
```

根据 Ascend SoC 型号选择最优通信方式：
- **A2/310P**：使用 AllGather 通信
- **A3/A5**：使用 AllToAll 通信（更高效的专家路由）

### 3. 运行时准备函数

```python
def prepare_hunyuan_fused_moe_runtime() -> None:
    world_size = torch.distributed.get_world_size()
    data_parallel_size = get_data_parallel_world_size()
    tensor_parallel_size = get_tensor_model_parallel_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    local_rank = get_world_group().local_rank

    _init_mc2_group_for_diffusion(...)

    _vllm_fc.ForwardContext.moe_comm_type = _select_moe_comm_method(...)
    _vllm_fc.ForwardContext.moe_comm_method = _MoECommMethods.get(...)
    _vllm_fc.ForwardContext.flash_comm_v1_enabled = False
```

该函数在扩散模型执行前被 `NPUOmniPlatform.prepare_diffusion_op_runtime()` 调用，完成：
1. 初始化 MC2 通信组
2. 将 MoE 通信配置注入 vLLM 的 `ForwardContext`

### 4. AscendHunyuanFusedMoE 模型类

```python
class AscendHunyuanFusedMoE(AscendSharedFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix
        self._init_hook_handle = self.register_forward_pre_hook(
            self._initialize_kernel_hook, with_kwargs=True
        )

    def _initialize_kernel_hook(self, module, args, kwargs) -> None:
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()
```

关键设计：
- 继承 `AscendSharedFusedMoE`（vllm_ascend 提供的共享 MoE 基类）
- 使用 **forward pre-hook** 延迟初始化量化权重，在第一次前向传播时自动处理，之后移除 hook
- `__del__` 方法负责清理 MC2 通信组资源

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `_init_mc2_group_for_diffusion()` | 函数 | 初始化 MC2 通信组 |
| `_select_moe_comm_method()` | 函数 | 按 SoC 型号选择通信方式 |
| `prepare_hunyuan_fused_moe_runtime()` | 函数 | 运行时初始化入口 |
| `AscendHunyuanFusedMoE` | 类 | Hunyuan MoE 的 Ascend 实现 |

## 与其他模块的关系

- **调用方**：`NPUOmniPlatform.prepare_diffusion_op_runtime()` 和 `NPUOmniPlatform.get_diffusion_model_impl_qualname()`
- **基类来源**：`vllm_ascend.ops.fused_moe.fused_moe.AscendSharedFusedMoE`
- **通信层**：`vllm_ascend.distributed.parallel_state`（MC2 通信组）
- **Omni 并行层**：`vllm_omni.diffusion.distributed.parallel_state`（数据并行信息）

## 总结

该文件是 NPU 平台最具特色的实现之一，展示了如何为特定硬件定制 MoE 算子。通过按 SoC 型号自动选择最优通信方式、延迟初始化量化权重、以及 MC2 通信组生命周期管理，使 HunyuanFusedMoE 能在不同型号的 Ascend NPU 上高效运行。
