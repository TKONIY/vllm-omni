# `main.py` — bench 子命令路由

## 文件概述

实现了 `vllm bench --omni` 子命令的路由逻辑，自动发现并注册所有继承自 `OmniBenchmarkSubcommandBase` 的基准测试子命令。

## 关键代码解析

```python
class OmniBenchmarkSubcommand(CLISubcommand):
    name = "bench"

    def subparser_init(self, subparsers):
        bench_parser = subparsers.add_parser(self.name, ...)
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        # 自动发现所有注册的基准测试子命令
        for cmd_cls in OmniBenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(cmd_cls.name, ...)
            cmd_subparser.add_argument("--omni", action="store_true")
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
        return bench_parser

def cmd_init() -> list[CLISubcommand]:
    return [OmniBenchmarkSubcommand()]
```

通过 `__subclasses__()` 机制，任何新的基准测试只需继承 `OmniBenchmarkSubcommandBase` 即可自动注册到 CLI 中。

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniBenchmarkSubcommand` | 类 | bench 子命令路由器 |
| `cmd_init()` | 函数 | 命令工厂方法，被 main.py 调用 |

## 与其他模块的关系

- 被 `cli/main.py` 通过 `cmd_init()` 注册
- 自动发现 `base.py` 的所有子类（如 `serve.py`）

## 总结

一个基于 Python 子类发现机制的命令路由器，使新增基准测试子命令无需修改路由代码。
