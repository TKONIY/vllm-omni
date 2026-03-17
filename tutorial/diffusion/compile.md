# `compile.py` — 区域编译加速

## 文件概述

`compile.py` 提供了区域编译（Regional Compilation）功能，用于对扩散模型中的重复模块块应用 `torch.compile` 加速。与全模型编译不同，区域编译仅对模型中指定的重复子模块（如 Transformer Block）进行编译，从而在编译开销与推理加速之间取得平衡。

## 关键代码解析

### `regionally_compile` 函数

```python
def regionally_compile(model: nn.Module, *compile_args: Any, **compile_kwargs: Any) -> nn.Module:
    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning("Regional compilation skipped because the model does not define `_repeated_blocks`.")
        return model

    has_compiled_region = False
    for submod in model.modules():
        if submod.__class__.__name__ in repeated_blocks:
            submod.compile(*compile_args, **compile_kwargs)
            has_compiled_region = True

    if not has_compiled_region:
        logger.warning(f"Regional compilation skipped because {repeated_blocks} classes are not found in the model.")

    return model
```

核心逻辑：
1. 从模型上读取 `_repeated_blocks` 属性，获取需要编译的子模块类名列表。
2. 遍历模型的所有子模块，若其类名在列表中，则调用 `submod.compile()` 进行编译。
3. 模型本身被原地修改并返回。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `regionally_compile` | 函数 | 对模型中 `_repeated_blocks` 指定的重复子模块应用 `torch.compile` |

## 与其他模块的关系

- 被 `worker/diffusion_model_runner.py` 中的 `DiffusionModelRunner._compile_transformer` 调用，在模型加载后对 transformer 子模块进行区域编译。
- 模型类需要定义 `_repeated_blocks` 属性（一个类名字符串列表）来指定哪些子模块需要编译。

## 总结

`compile.py` 实现了一种精细化的模型编译策略：仅编译模型中重复出现的 Transformer Block 子模块，避免了全模型编译的高开销，同时获得推理加速的收益。该机制通过约定式接口（`_repeated_blocks` 属性）实现，与模型代码解耦。
