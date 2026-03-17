# `serialization.py` — 序列化与反序列化框架

## 文件概述

该文件实现了 OmniConnector 系统的序列化框架，基于 `msgspec.msgpack` 构建。支持 `torch.Tensor`、`numpy.ndarray`、`PIL.Image`、`RequestOutput`、`CompletionOutput`、`OmniRequestOutput` 等复杂类型的编解码。

## 关键代码解析

### 1. 类型标记

```python
_TENSOR_MARKER = "__tensor__"
_NDARRAY_MARKER = "__ndarray__"
_PIL_IMAGE_MARKER = "__pil_image__"
```

使用字典中的特殊 key 标识自定义类型，在反序列化时据此重建原始对象。

### 2. OmniMsgpackEncoder — 编码器

```python
class OmniMsgpackEncoder:
    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self._enc_hook)

    def _enc_hook(self, obj):
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)
        if isinstance(obj, np.ndarray):
            return self._encode_ndarray(obj)
        if isinstance(obj, Image.Image):
            return self._encode_pil_image(obj)
        if isinstance(obj, RequestOutput):
            return self._encode_request_output(obj)
        if isinstance(obj, CompletionOutput):
            return self._encode_completion_output(obj)
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, slice):
            return (obj.start, obj.stop, obj.step)
```

Tensor 编码示例：

```python
def _encode_tensor(self, tensor):
    t = tensor.detach().contiguous().cpu()
    if t.dim() == 0:
        t = t.reshape(1)  # 处理标量 tensor
    t = t.view(torch.uint8)
    return {
        _TENSOR_MARKER: True,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": t.numpy().tobytes(),
    }
```

RequestOutput 编码的特殊处理：
- `RequestOutput` 不是 dataclass，需要手动提取属性
- 保留动态属性 `multimodal_output`
- 逐个编码 `CompletionOutput` 以保留 multimodal 数据

### 3. OmniMsgpackDecoder — 解码器

```python
class OmniMsgpackDecoder:
    def _post_process(self, obj):
        if isinstance(obj, dict):
            if obj.get(_TENSOR_MARKER):
                return self._decode_tensor(obj)
            if obj.get(_NDARRAY_MARKER):
                return self._decode_ndarray(obj)
            if obj.get(_PIL_IMAGE_MARKER):
                return self._decode_pil_image(obj)

            processed = {k: self._post_process(v) for k, v in obj.items()}

            # 检查顺序：先 OmniRequestOutput，再 RequestOutput
            if self._is_omni_request_output(processed):
                return self._decode_omni_request_output(processed)
            if _REQUEST_OUTPUT_KEYS.issubset(processed.keys()):
                return self._decode_request_output(processed)
```

解码器递归处理嵌套结构，通过 key 集合判断自动重建对象类型。

Tensor 解码：
```python
def _decode_tensor(self, obj):
    buffer = bytearray(data)
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    return arr.view(torch_dtype).reshape(shape)
```

### 4. 全局实例

```python
class OmniSerde:
    def __init__(self):
        self.encoder = OmniMsgpackEncoder()
        self.decoder = OmniMsgpackDecoder()

OmniSerializer = OmniSerde()  # 全局单例
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `OmniMsgpackEncoder` | class | 自定义 msgpack 编码器 |
| `OmniMsgpackDecoder` | class | 自定义 msgpack 解码器 |
| `OmniSerde` | class | 编解码器组合 |
| `OmniSerializer` | global instance | 全局序列化器单例 |
| `_encode_tensor()` | method | 编码 torch.Tensor |
| `_encode_ndarray()` | method | 编码 numpy.ndarray |
| `_encode_pil_image()` | method | 编码 PIL.Image |
| `_encode_request_output()` | method | 编码 vLLM RequestOutput |
| `_decode_tensor()` | method | 解码 torch.Tensor |
| `_decode_request_output()` | method | 解码 RequestOutput |
| `_decode_omni_request_output()` | method | 解码 OmniRequestOutput |

## 与其他模块的关系

- 被 `OmniConnectorBase.serialize_obj()` / `deserialize_obj()` 调用
- 被 `MooncakeTransferEngineConnector` 直接使用（对非原始类型序列化）
- 处理 vLLM 的 `RequestOutput` / `CompletionOutput` 和 vllm-omni 的 `OmniRequestOutput`

## 总结

`serialization.py` 是连接器系统的序列化中枢。基于 `msgspec.msgpack` 提供高性能编解码，通过自定义 hook 和类型标记机制支持 PyTorch tensor、NumPy 数组、PIL 图像和 vLLM 输出对象等复杂类型。注意当前尚未实现零拷贝优化（TODO）。
