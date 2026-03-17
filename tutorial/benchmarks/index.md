# benchmarks 模块索引

本模块为 vllm-omni 提供多模态（文本+音频）基准测试能力，扩展了 vLLM 原生的基准测试框架。

## 模块结构

```
benchmarks/
├── serve.py                              # 基准测试入口
├── data_modules/
│   ├── __init__.py                       # 空初始化文件
│   └── random_multi_modal_dataset.py     # 多模态随机数据集生成
├── metrics/
│   ├── __init__.py                       # 空初始化文件
│   └── metrics.py                        # 多模态指标计算与打印
└── patch/
    ├── __init__.py                       # 空初始化文件
    └── patch.py                          # 猴子补丁：注入自定义后端与基准测试逻辑
```

## 文档列表

| 文件 | 说明 |
|------|------|
| [serve.md](serve.md) | 基准测试主入口 |
| [data_modules/__init__.md](data_modules/__init__.md) | data_modules 包初始化 |
| [data_modules/random_multi_modal_dataset.md](data_modules/random_multi_modal_dataset.md) | 多模态随机数据集 |
| [metrics/__init__.md](metrics/__init__.md) | metrics 包初始化 |
| [metrics/metrics.md](metrics/metrics.md) | 基准测试指标体系 |
| [patch/__init__.md](patch/__init__.md) | patch 包初始化 |
| [patch/patch.md](patch/patch.md) | 猴子补丁与自定义后端 |

## 模块间关系

- `serve.py` 是顶层入口，调用 vLLM 的 `main_async`。
- `patch/patch.py` 通过猴子补丁替换 vLLM 的 `datasets.get_samples` 和 `serve.benchmark`，注入多模态支持。
- `data_modules/random_multi_modal_dataset.py` 提供音频/视频合成数据集。
- `metrics/metrics.py` 扩展指标体系，增加音频相关指标（TTFP、RTF、音频时长等）。
