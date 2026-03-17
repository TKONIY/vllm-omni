# `gguf.py` — GGUF 量化配置与线性方法

## 文件概述

`gguf.py` 实现了扩散 transformer 模型的 GGUF 量化支持。GGUF 是 llama.cpp 生态的量化格式，支持多种量化类型（Q4_0、Q5_1、Q8_0 等）。本文件提供了自定义的反量化 GEMM 路径和配置类。

## 关键代码解析

### 1. 反量化矩阵乘法

```python
def dequant_gemm_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
    return x @ weight.T
```

对于量化类型的权重，先调用 `ggml_dequantize` 还原为浮点张量，再执行标准矩阵乘法。非量化类型（如 F32、F16）直接计算。

### 2. 扩散模型专用线性方法

```python
class DiffusionGGUFLinearMethod(GGUFLinearMethod):
    def apply(self, layer, x, bias=None):
        shard_id = getattr(layer.qweight, "shard_id", None)
        if shard_id:
            # 分片反量化：对每个分片分别反量化后拼接
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(dequant_gemm_gguf(x, qweight[start:end, :offset].contiguous(), qweight_type))
            out = torch.cat(result, axis=-1)
        else:
            out = dequant_gemm_gguf(x, qweight, qweight_type)
```

关键差异在于分片处理：当权重被分片（如 QKV 融合投影的各部分可能使用不同量化类型时），需要逐分片反量化后拼接结果。

### 3. 自定义 GGUF 配置

```python
class _GGUFConfig(GGUFConfig):
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.unquantized_modules, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return DiffusionGGUFLinearMethod(self)
        return None
```

覆盖了 vLLM 的 `GGUFConfig`，使其返回扩散模型专用的 `DiffusionGGUFLinearMethod`。

### 4. 对外配置类

```python
class DiffusionGgufConfig(DiffusionQuantizationConfig):
    quant_config_cls = GGUFConfig

    def __init__(self, gguf_model=None, unquantized_modules=None):
        self.gguf_model = gguf_model
        self.unquantized_modules = unquantized_modules or []
        self._vllm_config = _GGUFConfig(unquantized_modules=self.unquantized_modules)
```

`gguf_model` 参数支持多种格式：本地文件路径、`repo_id/filename.gguf`、`repo_id:quant_type`。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `dequant_gemm_gguf` | 函数 | GGUF 反量化 + 矩阵乘法 |
| `DiffusionGGUFLinearMethod` | 类 | 扩散模型专用 GGUF 线性方法，支持分片反量化 |
| `_GGUFConfig` | 类 | 自定义 GGUF 配置，返回扩散专用线性方法 |
| `DiffusionGgufConfig` | 类 | GGUF 量化配置入口，管理 GGUF 模型路径和排除模块 |

## 与其他模块的关系

- **`base.py`**：`DiffusionGgufConfig` 继承 `DiffusionQuantizationConfig`。
- **`__init__.py`**：注册为 `"gguf"` 量化方法。
- **`model_loader/gguf_adapters/`**：GGUF 权重加载由 gguf_adapters 处理，本模块负责推理时的量化计算。
- **vLLM GGUF**：复用 `GGUFConfig`、`GGUFLinearMethod`、`ggml_dequantize` 等基础组件。

## 总结

`gguf.py` 将 GGUF 量化格式引入扩散模型推理。与 FP8 的在线量化不同，GGUF 使用预量化的权重文件，在推理时执行反量化 + GEMM 操作。分片反量化的设计确保了与 QKV 融合投影等复杂层结构的兼容性。
