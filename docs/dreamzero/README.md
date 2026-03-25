# DreamZero Pipeline 代码解读系列

DreamZero 在 vllm-omni 中的完整实现解读。覆盖架构设计、端到端执行流程、核心组件实现细节。

Draft PR: [#2162](https://github.com/vllm-project/vllm-omni/pull/2162)

## 目录

| 文档 | 内容 |
|------|------|
| [01-architecture.md](01-architecture.md) | 架构总览：DreamZero 在 vllm-omni 中的位置 |
| [02-e2e-flow.md](02-e2e-flow.md) | 端到端执行流程：从 WebSocket 请求到 action 输出 |
| [03-causal-wan-model.md](03-causal-wan-model.md) | CausalWanModel：40 层 DiT 的因果注意力与 KV Cache |
| [04-cfg-parallel.md](04-cfg-parallel.md) | CFG Parallel 适配：双输出模型的 all_gather + 本地 combine |
| [05-websocket-serving.md](05-websocket-serving.md) | WebSocket Serving：OpenPI 风格协议与会话管理 |
| [06-precision-testing.md](06-precision-testing.md) | 精度对齐与测试：bit-identical 验证方法 |
