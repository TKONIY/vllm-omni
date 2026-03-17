# `cuda_graph_decoder_wrapper.py` — CUDA Graph 解码器加速

## 文件概述

本文件实现了 `CUDAGraphDecoderWrapper`，用于对语音分词器解码器的前向传播进行 CUDA Graph 捕获和重放，从而减少 GPU kernel launch 的开销，提升推理吞吐量。

## 关键代码解析

### 1. 预定义捕获尺寸

```python
class CUDAGraphDecoderWrapper:
    DEFAULT_CAPTURE_SIZES = [2, 4, 8, 16, 25, 32, 50, 100, 150, 200, 250, 300]
```

覆盖常见的 codec 序列长度。

### 2. 预热与图捕获

```python
def warmup(self, device, dtype=torch.long):
    # 1. 预热运行（确保内存分配完成）
    for size in self.capture_sizes:
        dummy_codes = torch.zeros(1, self.num_quantizers, size, ...)
        _ = self.decoder(dummy_codes)
    torch.cuda.synchronize(device)
    # 2. 捕获计算图
    for size in self.capture_sizes:
        self._capture_graph_for_size(size, device, dtype)

def _capture_graph_for_size(self, size, device, dtype):
    static_input = torch.zeros(1, self.num_quantizers, size, ...)
    graph = CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = self.decoder(static_input)
    self.graphs[size] = graph
    self.static_inputs[size] = static_input
    self.static_outputs[size] = static_output
```

### 3. 解码（自动选择 Graph 或 Eager）

```python
def decode(self, codes):
    actual_size = codes.shape[-1]
    padded_size = self._get_padded_size(actual_size)  # 向上取整到捕获尺寸
    if padded_size is None:
        return self.decoder(codes)  # 超出范围，退化为 eager
    self.static_inputs[padded_size].zero_()
    self.static_inputs[padded_size][:, :, :actual_size] = codes
    self.graphs[padded_size].replay()
    return self.static_outputs[padded_size][..., :actual_output_len].clone()
```

### 4. 分块解码与 CUDA Graph

```python
def chunked_decode_with_cudagraph(self, codes, chunk_size=300, left_context_size=25):
    while start_index < total_len:
        codes_chunk = codes[..., start_index - context_size : end_index]
        wav_chunk = self.decode(codes_chunk)  # 每块使用 CUDA Graph
        wavs.append(wav_chunk[..., context_size * total_upsample :])
```

## 核心类/函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CUDAGraphDecoderWrapper` | 类 | CUDA Graph 包装器 |
| `warmup()` | 方法 | 预热和图捕获 |
| `decode()` | 方法 | 智能解码（Graph/Eager） |
| `chunked_decode_with_cudagraph()` | 方法 | 分块 Graph 解码 |

## 与其他模块的关系

- **被引用**: `qwen3_tts.py` 中为解码器创建 CUDA Graph 包装
- **包装**: SpeechTokenizer 解码器模块

## 总结

CUDA Graph 包装器通过预捕获固定大小的计算图来加速解码推理。核心策略：(1) 输入 pad 到预定义尺寸；(2) 图重放后裁剪输出到实际长度；(3) 超出预定义范围时自动退化为 eager 模式。适用于 batch_size=1 的场景。
