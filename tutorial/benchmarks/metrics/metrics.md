# `metrics.py` — 多模态基准测试指标体系

## 文件概述

该文件定义了 vllm-omni 的基准测试指标数据结构和计算逻辑。核心类 `MultiModalsBenchmarkMetrics` 继承 vLLM 的 `BenchmarkMetrics`，增加了音频相关的性能指标。`calculate_metrics` 函数负责从请求输出中汇总和计算所有指标，`print_metrics` 系列函数负责格式化输出。

## 关键代码解析

### 音频指标数据类

```python
@dataclass
class MultiModalsBenchmarkMetrics(BenchmarkMetrics):
    mean_audio_ttfp_ms: float = 0.0      # 音频首包时间（Time To First Packet）
    median_audio_ttfp_ms: float = 0.0
    total_audio_duration_s: float = 0.0   # 总音频时长
    total_audio_frames: int = 0           # 总音频帧数
    audio_throughput: float = 0.0         # 音频吞吐量（音频秒数/实际秒数）
    mean_audio_rtf: float = 0.0          # 实时因子（Real Time Factor）
    mean_audio_duration_s: float = 0.0    # 平均音频时长
    # ... 还包含 std、median、percentiles 变体
```

新增三组音频核心指标：
1. **TTFP (Time To First Packet)**: 从请求发出到收到第一个音频包的时间
2. **RTF (Real Time Factor)**: 生成时间与音频时长的比值，RTF < 1 表示实时
3. **Audio Duration**: 生成的音频时长统计

### 指标计算核心

```python
def calculate_metrics(input_requests, outputs, dur_s, tokenizer, ...) -> tuple:
    # 遍历所有输出，收集各项数据
    for i in range(len(outputs)):
        if outputs[i].success:
            audio_ttfps.append(getattr(outputs[i], "audio_ttfp", 0.0))
            audio_rtfs.append(getattr(outputs[i], "audio_rtf", 0.0))
            audio_duration.append(getattr(outputs[i], "audio_duration", 0.0))
            # ...

    # 计算峰值吞吐量和并发请求数
    tokens_per_second = np.zeros(duration_seconds)
    for i, output in enumerate(successful_outputs):
        token_times = [output.start_time + output.ttft]
        # ...

    # 构造最终指标对象
    metrics = MultiModalsBenchmarkMetrics(
        audio_throughput=sum(audio_duration) / dur_s,
        mean_audio_rtf=np.mean(audio_rtfs or 0),
        # ...
    )
```

该函数逻辑分为几个阶段：
1. 遍历所有请求输出，收集文本和音频指标原始数据
2. 计算 goodput（满足 SLO 的请求比例）
3. 按秒粒度统计 token 吞吐量和并发数，找到峰值
4. 汇总计算均值、中位数、百分位数等统计量
5. 调用 `print_metrics` 输出格式化结果

### 指标打印

```python
def print_metrics(task_type, selected_percentile_metrics, ...):
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    # ... 打印通用指标
    print_text_metrics(task_type, selected_percentile_metrics, metrics)
    if task_type == TaskType.GENERATION:
        print_audio_metrics(selected_percentile_metrics, metrics)
```

输出分为三个区域：总览、文本指标、音频指标，每个区域内通过 `process_one_metric` 统一处理均值/中位数/百分位数的打印。

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `MultiModalsBenchmarkMetrics` | 数据类 | 扩展 `BenchmarkMetrics`，增加音频指标字段 |
| `calculate_metrics(...)` | 函数 | 从请求输出计算所有基准测试指标 |
| `print_metrics(...)` | 函数 | 格式化打印基准测试结果 |
| `print_text_metrics(...)` | 函数 | 打印文本相关指标 |
| `print_audio_metrics(...)` | 函数 | 打印音频相关指标 |
| `process_one_metric(...)` | 函数 | 通用的单指标打印逻辑 |

## 与其他模块的关系

- **继承 vLLM**: `MultiModalsBenchmarkMetrics` 继承 `vllm.benchmarks.serve.BenchmarkMetrics`。
- **被 patch 模块调用**: `patch/patch.py` 中的 `benchmark` 函数调用 `calculate_metrics` 汇总指标。
- **依赖数据结构**: 使用 vLLM 的 `SampleRequest` 和 `RequestFuncOutput` 作为输入。

## 总结

该文件是 vllm-omni 基准测试的指标核心，在 vLLM 原有的文本指标（TTFT、TPOT、ITL、E2EL）基础上，增加了音频推理特有的指标体系（TTFP、RTF、音频时长/帧数/吞吐量），为多模态模型的性能评估提供了完整的量化框架。
