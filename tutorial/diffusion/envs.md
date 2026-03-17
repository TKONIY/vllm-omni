# `envs.py` — 环境变量与包检测

## 文件概述

`envs.py` 管理扩散模块的运行时环境变量，并提供包可用性检测功能。它采用惰性求值（lazy evaluation）机制访问环境变量，同时通过单例模式检测 Flash Attention 等关键依赖的可用性。该文件改编自 xDiT 项目。

## 关键代码解析

### 环境变量惰性求值

```python
environment_variables: dict[str, Callable[[], Any]] = {
    "MASTER_ADDR": lambda: os.getenv("MASTER_ADDR", ""),
    "MASTER_PORT": lambda: int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None,
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
}

def __getattr__(name):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

通过模块级 `__getattr__`，实现 `envs.MASTER_ADDR` 这种直接属性访问方式，每次访问时从环境变量中实时读取。

### PackagesEnvChecker — 包可用性检测

```python
class PackagesEnvChecker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def _check_flash_attn(self, packages_info) -> bool:
        # 检查平台是否为 CUDA
        # 排除不支持的 GPU（Turing/Tesla/T4）
        # 按优先级尝试 FA3 (fa3_fwd_interface) -> FA3 (flash_attn_interface) -> FA2 (flash_attn)
```

使用单例模式在首次访问时检测 Flash Attention 的可用性，支持 FA2 和 FA3 两个版本，并检查 GPU 硬件兼容性。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `environment_variables` | dict | 环境变量名到惰性求值函数的映射 |
| `PackagesEnvChecker` | 类 | 单例，检测 Flash Attention 等包的可用性 |
| `PACKAGES_CHECKER` | 实例 | `PackagesEnvChecker` 的全局单例 |

## 与其他模块的关系

- `MASTER_ADDR`、`MASTER_PORT`、`LOCAL_RANK` 等变量被 `worker/diffusion_worker.py` 用于分布式初始化。
- `PackagesEnvChecker` 的结果被注意力后端选择器使用，决定使用 Flash Attention 还是 PyTorch SDPA。

## 总结

`envs.py` 提供了两个核心功能：环境变量的惰性访问和依赖包的可用性检测。环境变量通过 Python 模块级 `__getattr__` 实现延迟求值，包检测通过单例模式保证仅执行一次。这些基础设施为分布式环境和硬件适配提供支撑。
