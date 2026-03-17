# `base.py` — GGUF 适配器基类与权重迭代器

## 文件概述

`base.py` 定义了 GGUF 适配器的抽象基类 `GGUFAdapter` 和通用的 GGUF 量化权重迭代器 `gguf_quant_weights_iterator`。适配器负责将 GGUF 文件中的键名映射到扩散模型的参数名。

## 关键代码解析

### 1. MappedTensor 数据类

```python
@dataclass
class MappedTensor:
    name: str                         # 目标参数名
    tensor: Any                       # 张量数据
    tensor_type: Any                  # GGUF 量化类型
    row_slice: slice | None = None    # 可选的行切片
    swap_scale_shift: bool = False    # 是否交换 scale/shift
```

描述一个映射后的张量，支持行切片和 scale/shift 交换等特殊操作。

### 2. GGUFAdapter 基类

```python
class GGUFAdapter(ABC):
    def __init__(self, gguf_file, model, source, od_config):
        self.gguf_file = gguf_file
        self.model = model
        self.source = source
        self.od_config = od_config

    @staticmethod
    def is_compatible(od_config, model, source) -> bool:
        return False  # 子类必须实现

    @abstractmethod
    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError
```

子类需要实现：
- `is_compatible()`：静态方法，判断此适配器是否适用于给定模型。
- `weights_iterator()`：生成器，产出 `(参数名, 张量)` 对。

### 3. GGUF 量化权重迭代器

```python
def gguf_quant_weights_iterator(gguf_file: str) -> Generator[tuple[str, torch.Tensor]]:
    reader = gguf.GGUFReader(gguf_file)

    # 第一遍：先产出所有量化类型信息
    for tensor in reader.tensors:
        if weight_type.name not in ("F32", "F16"):
            weight_type_name = name.replace("weight", "qweight_type")
            yield weight_type_name, torch.tensor(weight_type)

    # 第二遍：再产出所有权重数据
    for tensor in reader.tensors:
        if weight_type.name not in ("F32", "F16"):
            name = name.replace("weight", "qweight")
        # BF16 特殊处理：原始字节重新解释为 torch.bfloat16
        if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
            weight = weight.view(np.uint16)
            param = torch.tensor(weight).view(torch.bfloat16)
        else:
            param = torch.tensor(weight)
        yield name, param
```

关键设计：**必须先产出所有 `qweight_type`，再产出 `qweight`**。这是因为 packed 层的不同分片可能使用不同的量化类型，权重加载器需要先知道所有分片的类型才能正确初始化存储。

对于量化权重，键名中的 `weight` 被替换为 `qweight`，类型信息则使用 `qweight_type` 后缀。BF16 类型需要特殊的字节重解释处理。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `MappedTensor` | 数据类 | 描述映射后的张量元数据 |
| `GGUFAdapter` | 抽象类 | GGUF 适配器基类 |
| `is_compatible` | 静态方法 | 检查适配器是否兼容当前模型 |
| `weights_iterator` | 抽象方法 | 产出映射后的权重迭代器 |
| `gguf_quant_weights_iterator` | 函数 | 通用 GGUF 文件量化权重迭代器 |

## 与其他模块的关系

- **`flux2_klein.py`**：`Flux2KleinGGUFAdapter` 继承此基类并使用 `gguf_quant_weights_iterator`。
- **`z_image.py`**：`ZImageGGUFAdapter` 继承此基类并使用 `gguf_quant_weights_iterator`。
- **`__init__.py`**：工厂函数使用 `is_compatible` 进行匹配。
- **vLLM**：与 vLLM 的 GGUF 加载逻辑保持同步。

## 总结

`base.py` 建立了 GGUF 适配器的框架。`gguf_quant_weights_iterator` 处理了 GGUF 格式的底层细节（量化类型分离、BF16 字节解释等），子类只需关注键名映射逻辑。先产出类型再产出权重的两遍扫描设计是与 vLLM packed 层加载逻辑兼容的关键。
