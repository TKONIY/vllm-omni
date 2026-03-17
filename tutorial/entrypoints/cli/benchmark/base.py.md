# `base.py` — 基准测试子命令基类

## 文件概述

定义了 `OmniBenchmarkSubcommandBase` 抽象基类，为所有 Omni 基准测试子命令提供统一接口。

## 关键代码解析

```python
class OmniBenchmarkSubcommandBase(CLISubcommand):
    """vllm bench 子命令的基类"""
    help: str  # 子类必须定义帮助文本

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """子类必须实现: 添加 CLI 参数"""
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """子类必须实现: 执行基准测试"""
        raise NotImplementedError
```

所有基准测试子命令（如 `serve`）继承此基类并实现 `add_cli_args()` 和 `cmd()` 方法。`main.py` 通过 `__subclasses__()` 自动发现所有注册的子命令。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniBenchmarkSubcommandBase` | 抽象类 | 基准测试子命令的统一接口 |

## 与其他模块的关系

- 被 `serve.py` 中的 `OmniBenchmarkServingSubcommand` 继承
- 被 `main.py` 通过 `__subclasses__()` 自动发现和注册

## 总结

一个简洁的抽象基类，通过 Python 的子类自动发现机制实现基准测试子命令的插件式注册。
