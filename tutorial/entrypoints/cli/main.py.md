# `main.py` — CLI 入口点

## 文件概述

vLLM-Omni 的 CLI 主入口点，拦截 `vllm` 命令并根据是否存在 `--omni` 标志决定执行路径。不带 `--omni` 时直接委托给原生 vLLM CLI，带 `--omni` 时构建 Omni 专用的命令解析器。

## 关键代码解析

```python
def main():
    if "--omni" not in sys.argv:
        # 无 --omni 标志，直接使用 vLLM 原生 CLI
        from vllm.entrypoints.cli.main import main as vllm_main
        vllm_main()
        return
    else:
        # 构建 Omni CLI 解析器
        CMD_MODULES = [
            vllm_omni.entrypoints.cli.serve,
            vllm_omni.entrypoints.cli.benchmark.main,
        ]
        parser = FlexibleArgumentParser(description="vLLM OMNI CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")

        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)

        args = parser.parse_args()
        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
```

设计要点：
1. 通过检查 `sys.argv` 中的 `--omni` 标志实现零侵入式拦截
2. 各子命令模块通过 `cmd_init()` 工厂方法注册自己
3. 每个命令提供 `subparser_init()` 初始化参数、`validate()` 验证参数、`cmd()` 执行逻辑

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `main()` | 函数 | CLI 主入口，决定 vLLM 原生或 Omni 模式 |

## 与其他模块的关系

- 不带 `--omni` 时委托给 `vllm.entrypoints.cli.main`
- 注册 `serve.py` 和 `benchmark/main.py` 两个命令模块
- 是 `pyproject.toml` 中 `[project.scripts]` 的入口点

## 总结

一个精简的 CLI 路由器，通过 `--omni` 标志实现 vLLM 原生命令和 Omni 扩展命令的无缝切换，保持了与上游 vLLM 的兼容性。
